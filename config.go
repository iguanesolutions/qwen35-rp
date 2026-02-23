package main

import (
	"errors"
	"flag"
	"log/slog"
	"os"
	"strconv"
)

type Config struct {
	Listen              string
	Port                int
	Target              string
	LogLevel            string
	ServedModelName     string
	ThinkingModelName   string
	NoThinkingModelName string
}

func (c Config) Validate() error {
	if c.Listen == "" {
		return errors.New("listen address cannot be empty")
	}
	if c.Port <= 1024 || c.Port > 65535 {
		return errors.New("port must be a positive integer between 1024 and 65535")
	}
	if c.Target == "" {
		return errors.New("target cannot be empty")
	}
	if c.LogLevel == "" {
		return errors.New("log level cannot be empty")
	}
	if c.ServedModelName == "" {
		return errors.New("served model name cannot be empty")
	}
	if c.ThinkingModelName == "" {
		return errors.New("thinking model name cannot be empty")
	}
	if c.NoThinkingModelName == "" {
		return errors.New("no-thinking model name cannot be empty")
	}
	return nil
}

func LoadConfig() (Config, error) {
	var cfg Config

	listen := flag.String("listen", "0.0.0.0", "IP address to listen on")
	port := flag.Int("port", 9000, "Port to listen on")
	target := flag.String("target", "http://127.0.0.1:8000", "Backend target, default is for a local vLLM")
	loglevel := flag.String("loglevel", slog.LevelInfo.String(), "Log level (DEBUG, INFO, WARN, ERROR)")
	servedModel := flag.String("served-model", "", "Name of the served model")
	thinkingModel := flag.String("thinking-model", "", "Name of the thinking model")
	noThinkingModel := flag.String("no-thinking-model", "", "Name of the no-thinking model")

	flag.Parse()

	cfg.Listen = getEnvOrFlag(*listen, "KIMIRP_LISTEN", "0.0.0.0")
	cfg.Port = getEnvOrFlagInt(*port, "KIMIRP_PORT", 9000)
	cfg.Target = getEnvOrFlag(*target, "KIMIRP_TARGET", "http://127.0.0.1:8000")
	cfg.LogLevel = getEnvOrFlag(*loglevel, "KIMIRP_LOGLEVEL", slog.LevelInfo.String())
	cfg.ServedModelName = getEnvOrFlag(*servedModel, "KIMIRP_SERVED_MODEL_NAME", "")
	cfg.ThinkingModelName = getEnvOrFlag(*thinkingModel, "KIMIRP_THINKING_MODEL_NAME", "")
	cfg.NoThinkingModelName = getEnvOrFlag(*noThinkingModel, "KIMIRP_NO_THINKING_MODEL_NAME", "")

	return cfg, cfg.Validate()
}

func getEnvOrFlag(flagVal string, envName string, defaultVal string) string {
	if envVal := os.Getenv(envName); envVal != "" {
		return envVal
	}
	if flagVal != "" {
		return flagVal
	}
	return defaultVal
}

func getEnvOrFlagInt(flagVal int, envName string, defaultVal int) int {
	if envVal := os.Getenv(envName); envVal != "" {
		if intVal, err := strconv.Atoi(envVal); err == nil {
			return intVal
		}
	}
	if flagVal != defaultVal {
		return flagVal
	}
	return defaultVal
}
