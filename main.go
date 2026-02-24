package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"sync/atomic"
	"time"

	"github.com/hekmon/httplog/v3"
	autoslog "github.com/iguanesolutions/auto-slog/v2"
	sysd "github.com/iguanesolutions/go-systemd/v6"
	sysdnotify "github.com/iguanesolutions/go-systemd/v6/notify"
)

const (
	stopTimeout = 3 * time.Minute
)

var (
	logger           *slog.Logger
	modifiedRequests atomic.Int64
)

func main() {
	cfg, err := LoadConfig()
	if err != nil {
		log.Fatalf("load config: %s\n", err)
	}

	// Init
	logger = autoslog.NewLogger(slog.HandlerOptions{
		AddSource: true,
		Level:     parseLogLevel(cfg.LogLevel),
	})
	backendURL, err := url.Parse(cfg.Target)
	if err != nil {
		logger.Error("failed to parse backend URL", slog.Any("error", err))
		os.Exit(1)
	}

	// Define HTTP handlers and middleware
	httplogger := httplog.New(logger, &httplog.Config{
		RequestDumpLogLevel:  COMPLETE,
		ResponseDumpLogLevel: COMPLETE,
	})
	http.HandleFunc("/", httplogger.LogFunc(proxy(backendURL,
		cfg.ServedModelName, cfg.ThinkingModelName, cfg.NoThinkingModelName)))

	// Prepare HTTP server and clean stop
	server := &http.Server{Addr: fmt.Sprintf("%s:%d", cfg.Listen, cfg.Port)}
	signalStopCtx, signalStopCtxCancel := signal.NotifyContext(context.Background(), os.Interrupt, os.Interrupt)
	defer signalStopCtxCancel()
	go cleanStop(signalStopCtx, server)

	// Handle systemd if needed
	if invocationID, sysdStarted := sysd.GetInvocationID(); sysdStarted {
		logger.Info("systemd detected, activating systemd integration",
			slog.String("invocation_id", invocationID),
		)
		go systemdIntegration(signalStopCtx, httplogger)
	} else {
		logger.Debug("systemd not detected, skipping systemd integration")
	}

	// Start server
	logger.Info("starting reverse proxy server",
		slog.String("listen", cfg.Listen),
		slog.Int("port", cfg.Port),
		slog.String("target", backendURL.String()),
	)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		logger.Error("failed to start HTTP server", "err", err)
		os.Exit(1)
	}
}

func systemdIntegration(signalStopCtx context.Context, httplogger *httplog.Logger) {
	var err error
	if err = sysdnotify.Ready(); err != nil {
		logger.Error("failed to send systemd ready notification", "err", err)
	}
	sysdUpdateTicker := time.NewTicker(time.Minute)
	defer sysdUpdateTicker.Stop()
	for {
		select {
		case <-sysdUpdateTicker.C:
			logger.Debug("sending systemd status notification")
			if err = sysdnotify.Status(fmt.Sprintf("Modified %d requests on the %d proxified",
				modifiedRequests.Load(),
				httplogger.TotalRequests(),
			)); err != nil {
				logger.Error("failed to send systemd status notification", "err", err)
			}
		case <-signalStopCtx.Done():
			if err = sysdnotify.Stopping(); err != nil {
				logger.Error("failed to send systemd stopping notification", "err", err)
			}
			return
		}
	}
}

func cleanStop(signalStopCtx context.Context, server *http.Server) {
	<-signalStopCtx.Done()
	logger.Info("shutting down HTTP server...",
		slog.Duration("grace_period", stopTimeout),
	)
	ctx, cancel := context.WithTimeout(context.Background(), stopTimeout)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		logger.Error("failed to shutdown HTTP server properly", "err", err)
	}
}
