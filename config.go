package main

import (
	"errors"
	"flag"
	"log/slog"
	"os"
	"strconv"
	"strings"
)

// COMPLETE is a log level more verbose than DEBUG for complete request/response dumps
const COMPLETE = slog.LevelDebug - 4
const COMPLETE_LEVEL = "COMPLETE"

type Config struct {
	Listen                 string
	Port                   int
	Target                 string
	LogLevel               string
	ServedModelName        string
	ThinkingGeneralModel   string
	ThinkingCodingModel    string
	InstructGeneralModel   string
	InstructReasoningModel string
	EnforceSamplingParams  bool
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
	if c.ThinkingGeneralModel == "" {
		return errors.New("thinking-general model name cannot be empty")
	}
	if c.ThinkingCodingModel == "" {
		return errors.New("thinking-coding model name cannot be empty")
	}
	if c.InstructGeneralModel == "" {
		return errors.New("instruct-general model name cannot be empty")
	}
	if c.InstructReasoningModel == "" {
		return errors.New("instruct-reasoning model name cannot be empty")
	}
	return nil
}

func LoadConfig() (Config, error) {
	var cfg Config

	listen := flag.String("listen", "0.0.0.0", "IP address to listen on")
	port := flag.Int("port", 9000, "Port to listen on")
	target := flag.String("target", "http://127.0.0.1:8000", "Backend target, default is for a local vLLM")
	loglevel := flag.String("loglevel", slog.LevelInfo.String(), "Log level (COMPLETE, DEBUG, INFO, WARN, ERROR)")
	servedModel := flag.String("served-model", "", "Name of the served model")
	thinkingGeneral := flag.String("thinking-general", "qwen3.5-thinking-general", "Name of the thinking-general model")
	thinkingCoding := flag.String("thinking-coding", "qwen3.5-thinking-coding", "Name of the thinking-coding model")
	instructGeneral := flag.String("instruct-general", "qwen3.5-instruct-general", "Name of the instruct-general model")
	instructReasoning := flag.String("instruct-reasoning", "qwen3.5-instruct-reasoning", "Name of the instruct-reasoning model")
	enforceSampling := flag.Bool("enforce-sampling-params", false, "Enforce sampling parameters, overriding client-provided values")

	flag.Parse()

	cfg.Listen = getEnvOrFlag(*listen, "QWEN35RP_LISTEN")
	cfg.Port = getEnvOrFlagInt(*port, "QWEN35RP_PORT")
	cfg.Target = getEnvOrFlag(*target, "QWEN35RP_TARGET")
	cfg.LogLevel = getEnvOrFlag(*loglevel, "QWEN35RP_LOGLEVEL")
	cfg.ServedModelName = getEnvOrFlag(*servedModel, "QWEN35RP_SERVED_MODEL_NAME")
	cfg.ThinkingGeneralModel = getEnvOrFlag(*thinkingGeneral, "QWEN35RP_THINKING_GENERAL_MODEL")
	cfg.ThinkingCodingModel = getEnvOrFlag(*thinkingCoding, "QWEN35RP_THINKING_CODING_MODEL")
	cfg.InstructGeneralModel = getEnvOrFlag(*instructGeneral, "QWEN35RP_INSTRUCT_GENERAL_MODEL")
	cfg.InstructReasoningModel = getEnvOrFlag(*instructReasoning, "QWEN35RP_INSTRUCT_REASONING_MODEL")
	cfg.EnforceSamplingParams = getEnvOrFlagBool(*enforceSampling, "QWEN35RP_ENFORCE_SAMPLING_PARAMS")

	return cfg, cfg.Validate()
}

func getEnvOrFlag(flagVal string, envName string) string {
	if envVal, exists := os.LookupEnv(envName); exists {
		return envVal
	}
	return flagVal
}

func getEnvOrFlagInt(flagVal int, envName string) int {
	if envVal, exists := os.LookupEnv(envName); exists {
		if intVal, err := strconv.Atoi(envVal); err == nil {
			return intVal
		}
	}
	return flagVal
}

func getEnvOrFlagBool(flagVal bool, envName string) bool {
	if envVal, exists := os.LookupEnv(envName); exists {
		if boolVal, err := strconv.ParseBool(envVal); err == nil {
			return boolVal
		}
	}
	return flagVal
}

// parseLogLevel parses a log level string, including the COMPLETE level
func parseLogLevel(levelStr string) slog.Level {
	switch strings.ToUpper(levelStr) {
	case COMPLETE_LEVEL:
		return COMPLETE
	case "DEBUG":
		return slog.LevelDebug
	case "INFO":
		return slog.LevelInfo
	case "WARN":
		return slog.LevelWarn
	case "ERROR":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
