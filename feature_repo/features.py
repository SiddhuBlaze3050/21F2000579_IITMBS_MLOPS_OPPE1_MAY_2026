from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float64, Int64, String

# Define entity
stock_entity = Entity(
    name="stock_name",
    value_type=ValueType.STRING,
    description="Stock identifier name",
)

# Define FileSource pointing to v0 feature parquet (Corrected Path!)
v0_source = FileSource(
    name="stock_v0_source",
    path="data/stock_features_v0.parquet",
    timestamp_field="timestamp",
)

# Define FeatureView for rolling metrics
stock_rolling_feature_view = FeatureView(
    name="stock_rolling_features",
    entities=[stock_entity],
    ttl=timedelta(days=3650), # Long TTL for historical coverage
    schema=[
        Field(name="rolling_avg_10", dtype=Float64),
        Field(name="volume_sum_10", dtype=Float64),
    ],
    online=True,
    source=v0_source,
)
