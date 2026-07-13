# TypeScript Binding Snippet

```typescript
import { fitModel, predictModel } from "@innovate/kernel";

const model = await fitModel("bass", {
  time: [1, 2, 3, 4],
  adoption: [3, 8, 15, 25],
});

const predictions = await predictModel(model, {
  horizon: 6,
  payload: { schema_version: "1.0" },
});
```
