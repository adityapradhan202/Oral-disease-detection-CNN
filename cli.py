import typer
from predict import predict_image
import os

app = typer.Typer()

@app.command()
def cli_predict(image_path:str):
    """CLI command to detect oral medical problems"""
    if os.path.exists(image_path):
        try:
            print("-> This will take a few seconds. Please wait!")
            predict_image(img_path=image_path)
        except Exception as e:
            print(f"Some exception occured: {e}")
    else:
        print("Provided path is not a valid file!")

if __name__ == "__main__":
    app()