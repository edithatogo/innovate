# Go Binding Snippet

```go
package main

import (
	"context"

	innovate "github.com/edithatogo/innovate/bindings/go"
)

func main() {
	ctx := context.Background()
	model, _ := innovate.FitModel(ctx, "bass", innovate.TablePayload{
		"time":     []float64{1, 2, 3, 4},
		"adoption": []float64{3, 8, 15, 25},
	})

	_, _ = innovate.PredictModel(ctx, model, innovate.PredictOptions{
		Horizon: 6,
		Payload: map[string]string{"schema_version": "1.0"},
	})
}
```
