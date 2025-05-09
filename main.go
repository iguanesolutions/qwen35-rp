package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"sync/atomic"
	"time"

	"github.com/hekmon/httplog/v2"
	autoslog "github.com/iguanesolutions/auto-slog"
	sysd "github.com/iguanesolutions/go-systemd/v5"
	sysdnotify "github.com/iguanesolutions/go-systemd/v5/notify"
)

const (
	stopTimeout = 3 * time.Minute
)

var (
	logger           *slog.Logger
	modifiedRequests atomic.Int64
	// overrided with build script
	Version = "dev"
)

func main() {
	// Flags
	listen := flag.String("listen", "0.0.0.0", "IP address to listen on")
	port := flag.Int("port", 9000, "Port to listen on")
	target := flag.String("target", "http://127.0.0.1:4000/v1", "Backend target, default is for a local vLLM")
	loglevel := flag.String("loglevel", slog.LevelInfo.String(), fmt.Sprintf("Valid log levels: %s, %s, %s, %s",
		slog.LevelDebug, slog.LevelInfo, slog.LevelWarn, slog.LevelError))
	version := flag.Bool("version", false, "Print version and exit")
	flag.Parse()

	// Special case
	if *version {
		fmt.Println("Version:", Version)
		os.Exit(0)
	}

	// Init
	logger = autoslog.NewLogger(autoslog.LogLevel(*loglevel))
	backend, err := url.Parse(*target)
	if err != nil {
		logger.Error("failed to parse backend URL", slog.Any("error", err))
		os.Exit(1)
	}

	// Define HTTP handlers and middleware
	httplogger := httplog.New(logger)
	http.HandleFunc("/", httplogger.LogFunc(proxy(backend)))

	// Prepare HTTP server and clean stop
	server := &http.Server{Addr: fmt.Sprintf("%s:%d", *listen, *port)}
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
		slog.String("listen", *listen),
		slog.Int("port", *port),
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
