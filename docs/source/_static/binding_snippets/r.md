# R Binding Snippet

```r
library(innovate.R)

model <- fit_model(
  model = "bass",
  data = data.frame(time = 1:4, adoption = c(3, 8, 15, 25))
)

predictions <- predict_model(
  model,
  horizon = 6,
  payload = list(schema_version = "1.0")
)
```
