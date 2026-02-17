from datetime import datetime
import json
from pathlib import Path
import sys

import great_expectations as gx
from loguru import logger
import pandas as pd
import typer

from data_processing.check_s3 import wait_for_s3_object
from iris_project.config import S3_BUCKET, S3_SILVER_PREFIX, VALIDATION_REPORTS_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: str = f"s3://{S3_BUCKET}/{S3_SILVER_PREFIX}/iris.parquet",
    log_output: Path = VALIDATION_REPORTS_DIR
    / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
):
    """Data Validation - Validate Silver dataset using Great Expectations."""
    logger.info(f"Loading Silver dataset from: {input_path}")
    wait_for_s3_object(S3_BUCKET, f"{S3_SILVER_PREFIX}/iris.parquet", timeout=60)

    df = pd.read_parquet(input_path, storage_options={"anon": False})
    if df.empty:
        logger.error("Input DataFrame is empty. Exiting validation.")
        sys.exit(1)

    data_source_name = "iris"
    data_asset_name = "iris_asset"
    batch_definition_name = "iris_batch"
    expectation_suite_name = "iris_suite"

    context = gx.get_context()
    data_source = context.data_sources.add_pandas(data_source_name)
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)

    batch_parameters = {"dataframe": df}
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    suite = gx.ExpectationSuite(name=expectation_suite_name)

    # Not null checks for all measurement columns
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    # Measurement ranges [0, 10]
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=0, max_value=10)
        )

    # Species values
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="species", value_set=["setosa", "versicolor", "virginica"]
        )
    )

    # Row count between 100 and 200
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=100, max_value=200)
    )

    context.suites.add_or_update(suite)

    logger.info("Running Great Expectations checks...")
    validation_results = batch.validate(suite)
    json_result = validation_results.to_json_dict()
    logger.info(validation_results.statistics)
    logger.info("Great Expectations checks complete.")

    log_output.parent.mkdir(parents=True, exist_ok=True)
    with open(log_output, "w") as f:
        json.dump(json_result, f, indent=4)
    logger.info(f"Validation log saved at: {log_output}")

    if not validation_results.success:
        logger.error("Data validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    app()
