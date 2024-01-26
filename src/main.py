import argparse
import yeadon
import uvicorn

from fastapi import FastAPI
from src.im2meas import *
from src.biomake.biomake_models import *
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

class YeadonModelRequest(BaseModel):
    impath_front: str
    impath_pike: str
    impath_r_tuck: str
    impath_side: str
    impath_tuck: str
    rotation: int
    mass: float
    calibration: int
    distance: int
    luminosity: int

@app.post("/process_yeadon_model/")
async def process_yeadon_model(request_data: YeadonModelRequest):
    try:
        yea = YeadonModel(request_data.impath_front, request_data.impath_pike, request_data.impath_r_tuck,
                          request_data.impath_side, request_data.impath_tuck,
                          request_data.rotation, request_data.mass, request_data.calibration,
                          request_data.distance, request_data.luminosity)
        bioModOptions = "src/biomake/tech_opt.yml"
        name = f"{request_data.impath_front.split('/')[-1].split('_')[0]}"
        human = yeadon.Human(f"{name}.txt")
        BioHuman, human_options, segments_options = parse_biomod_options(bioModOptions)
        biohuman = BioHuman(human, **human_options, **segments_options)
        return {"biomod": str(biohuman)}
    except Exception as e:
        return {"error_message": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Im2meas")

    parser.add_argument("front_img", type=str, help="Path to the front image")
    parser.add_argument("pike_img", type=str, help="Path to the pike image")
    parser.add_argument("right_tuck_img", type=str, help="Path to the right tuck image")
    parser.add_argument("side_img", type=str, help="Path to the side image")
    parser.add_argument("tuck_img", type=str, help="Path to the tuck image")
    # This is used because some phones rotate the images taken from the app, so set to 1 if you want to rotate it back to the original state
    parser.add_argument("--rotation", type=int, default=1, help="Enter 1 if you need to rotate the images")

    parser.add_argument("-m", "--mass", type=float, default=0, help="Enter the mass of the person")
    # Used if you want to use the calibration because it can be wrong
    parser.add_argument("-c", "--calibration", type=int, default=0, help="Enter 1 if you want to calibrate the images")
    # Used to change the distance between the camera and the wall just in case
    parser.add_argument("--distance", type=int, default=350, help="Enter the distance between the camera and the wall")
    parser.add_argument("-l", "--luminosity", type=int, default=0, help="Enter 1 if you want to increase the luminosity of the images")
    args = parser.parse_args()

    yeadon = process_yeadon_model(args.front_img, args.pike_img, args.right_tuck_img, args.side_img, args.tuck_img, args.rotation, args.mass, args.calibration, args.distance, args.luminosity)

