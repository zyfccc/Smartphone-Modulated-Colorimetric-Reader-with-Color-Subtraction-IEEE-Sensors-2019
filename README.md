# Smartphone Modulated Colorimetric Reader with Color Subtraction (IEEE Sensors 2019)

Source codes and 3D models for the Smartphone Modulated Colorimetric Reader with Color subtraction work.

These resources can be helpful for high-quality smartphones related researches.


## Environments
* Python 2.7
* Tensorflow 1.4.0
* Using AutoCAD to open and edit .dwg 3D model files
* Using FreeCAD to open and edit .FCStd1 3D model files


## 3D Model
This repo contains in total seven 3D models and their source files that compose the Smartphone Modulated Colorimetric Reader (SMCR).

All the parts can be 3D-printed with proper printer setup.


### Light box connector
This is the conenctor part that should be glued onto a commercial smartphone case to join with the light box. The camera and flashlight of the smartphone should be centered at the opening window of the connector. It should be 3D-printed in black material to prevent ambient light leakage.

### Light box
This is the part that blocks the ambient light and normalizes the light condition for optical assays. It should be 3D-printed in black material to prevent ambient light leakage. 

### Light source adaptor
This is the light source adaptor component. Here, we simply use this block to prevent ambient light from entering the light box as an example. Other light sources can be designed to provide light sources required for the optical assays. It should be 3D-printed in black material to prevent ambient light leakage. 

### Assay platform adaptor
This is the assay platform adaptor component. Here, we designed a pH test strip adaptor as an example. With this adaptor, pH test strips can be simply inserted into the light box from the side. It should be 3D-printed in black material to prevent ambient light leakage.

### Bottom panel and bottom panel cover
This is the bottom panel that closes the light box. The bottom panel could be 3D-printed in white material to ensure light reflectance while the bottom panel cover should be 3D-printed in black material to prevent ambient light leakage. A white paper could be used on top of the bottom panel to maximizes uniformity.


## Datasets
This repo includes three unique and challenging image datasets with in total 174 carefully labelled images. 

In the `training` folder are the pH test strips images produced with SMCR using transparent buffer pH solutions for the assay calibration. 

In the other two folders `blue` and `red` are the pH test strips images with SMCR and solution color catcher (SCC) using blue and red dyed pH buffer solutions, respectively. They are used for color subtration performance evaluation. 

The background images are provided with each image. In the `tags.json` files under each folder stores the bounding boxes of color areas and the true pH values of the solutions measured by a lab-based pH meter.


## Validation

Run `python color_subtraction.py` to train and validate the color subtraction model proposed in the paper.


## Publication

Please cite our paper if you find this work helpful:

https://ieeexplore.ieee.org/document/8956565



```
@inproceedings{zhao2019smartphone,
  title={Smartphone Modulated Colorimetric Reader with Color Subtraction},
  author={Zhao, Y and Choi, SY and Lou-Franco, J and Nelis, JLD and Zhou, H and Cao, C and Campbell, K and Elliott, C and Rafferty, K},
  booktitle={2019 IEEE SENSORS},
  pages={1--4},
  year={2019},
  organization={IEEE}
}
```

Cheers!

Yunfeng


## Acknowledgment
This project has received funding from the European Union’s Horizon 2020 research and innovation program under the Marie-Sklodowska-Curie grant agreement No 720325, FoodSmartphone.