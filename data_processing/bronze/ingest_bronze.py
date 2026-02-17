from loguru import logger
import pandas as pd
import typer

from iris_project.config import S3_BRONZE_PREFIX, S3_BUCKET

app = typer.Typer()


@app.command()
def main(
    input_path: str = f"s3://{S3_BUCKET}/{S3_BRONZE_PREFIX}/iris.csv",
    output_path: str = f"s3://{S3_BUCKET}/{S3_BRONZE_PREFIX}/iris.parquet",
):
    logger.info("Downloading Iris dataset...")

    df = pd.read_csv(input_path, storage_options={"anon": False})
    df["ingestion_time"] = pd.Timestamp.now()
    df.to_parquet(output_path, index=False, storage_options={"anon": False})

    logger.success(f"Dataset saved on {output_path}.")
    print(df.head())


if __name__ == "__main__":
    app()
