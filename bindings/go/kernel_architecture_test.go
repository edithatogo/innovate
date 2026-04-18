package innovate_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	innovate "github.com/edithatogo/innovate/bindings/go"
)

func TestKernelModuleArchitecture(t *testing.T) {
	t.Helper()

	if got := innovate.ModuleRoot(); !filepath.IsAbs(got) || !strings.HasSuffix(got, filepath.FromSlash("bindings/go")) {
		t.Fatalf("module root = %q, want an absolute path ending in %q", got, filepath.FromSlash("bindings/go"))
	}

	if got := innovate.BridgeScriptPath(); !filepath.IsAbs(got) || !strings.HasSuffix(got, filepath.FromSlash("bindings/go/inst/python/kernel_bridge.py")) {
		t.Fatalf("bridge script path = %q, want an absolute path ending in %q", got, filepath.FromSlash("bindings/go/inst/python/kernel_bridge.py"))
	}

	if _, err := os.Stat(innovate.BridgeScriptPath()); err != nil {
		t.Fatalf("bridge script must exist: %v", err)
	}

	if got := innovate.KernelSchemaVersion(); got != "1.0" {
		t.Fatalf("kernel schema version = %q, want %q", got, "1.0")
	}
}
