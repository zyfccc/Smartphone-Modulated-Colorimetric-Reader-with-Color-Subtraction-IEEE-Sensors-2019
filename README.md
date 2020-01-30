# Smartphone Modulated Colorimetric Reader with Color Subtraction (IEEE Sensors 2019)

Source codes, image datasets, and 3D models for the Smartphone Modulated Colorimetric Reader with Color subtraction work.

These resources can be helpful for high-quality smartphones related researches.


## Environments

* Python 2.7
* Tensorflow 1.4.0
* Using AutoCAD to open and edit .dwg 3D model files
* Using FreeCAD to open and edit .FCStd1 3D model files


## 3D Model

This repo contains in total seven 3D models and their source files that compose the Smartphone Modulated Colorimetric Reader (SMCR).

All the parts can be 3D-printed with proper printer setups.


<p align="center">
<img src="https://github.com/zyfccc/Smartphone-Modulated-Colorimetric-Reader-with-Color-Subtraction-IEEE-Sensors-2019/blob/master/resources/demo2.jpg" height="220">
<img src="https://github.com/zyfccc/Smartphone-Modulated-Colorimetric-Reader-with-Color-Subtraction-IEEE-Sensors-2019/blob/master/resources/demo.jpg" height="220">
</p>


### Light box connector

<p align="center">
<img src="https://github.com/zyfccc/Smartphone-Modulated-Colorimetric-Reader-with-Color-Subtraction-IEEE-Sensors-2019/blob/master/resources/smartphone_case_connector.jpg" height="120">
</p>

This is the conenctor part that should be glued onto a commercial smartphone case to enable docking with the light box. The camera and flashlight of the smartphone should be centered at the opening window of the connector. It should be 3D-printed in black material to prevent ambient light leakage.

### Light box

<p align="center">
<img src="https://github.com/zyfccc/Smartphone-Modulated-Colorimetric-Reader-with-Color-Subtraction-IEEE-Sensors-2019/blob/master/resources/light_box.jpg" height="220">
</p>

This is the part that creates a normalized lighting condition for the optical assays. It should be 3D-printed using black material to prevent ambient light leakage. 

### Light source adaptor

<p align="center">
<img src="https://github.com/zyfccc/Smartphone-Modulated-Colorimetric-Reader-with-Color-Subtraction-IEEE-Sensors-2019/blob/master/resources/light_source_adaptor.jpg" height="70">
</p>

This is the light source adaptor component. Here, we simply use a block to represent a potential light source and prevent the ambient light from entering the light box as an example. Other light sources can be designed to provide light sources required for different optical assays. It should be 3D-printed using black material to prevent ambient light leakage. 

### Assay platform adaptor

<p align="center">
<img src="https://github.com/zyfccc/Smartphone-Modulated-Colorimetric-Reader-with-Color-Subtraction-IEEE-Sensors-2019/blob/master/resources/ph_adaptor.jpg" height="150">
</p>

This is the assay platform adaptor component. Here, we designed a pH test strip adaptor as an example. With this adaptor, pH test strips can be simply inserted into the light box from the front of the light box. It should be 3D-printed using black material to prevent ambient light leakage.

### Bottom panel and bottom panel cover

<p align="center">
<img src="https://github.com/zyfccc/Smartphone-Modulated-Colorimetric-Reader-with-Color-Subtraction-IEEE-Sensors-2019/blob/master/resources/bottom_panel.jpg" height="100">
</p>

This is the bottom panel that closes the light box from the bottom. The bottom panel could be 3D-printed using white material to ensure a maximized internal light reflectance from the bottom. The bottom panel cover should be 3D-printed using black material to prevent ambient light leakage. A white paper could be used on top of the bottom panel to maximizes the light reflectance uniformity.


## Image Datasets

This repo includes three unique and challenging image datasets with a total of 174 carefully labelled images taken by an iPhone6P with the proposed SMCR, smartphone flashlight, and pH test strips produced from both transparent and colored solutions. 

In the `training` folder are the pH test strips images produced with SMCR using transparent pH buffer solutions for the assay calibration. 


<p align="center">
<img src="https://github.com/zyfccc/Smartphone-Modulated-Colorimetric-Reader-with-Color-Subtraction-IEEE-Sensors-2019/blob/master/resources/blue.jpg" height="220">
<img src="https://github.com/zyfccc/Smartphone-Modulated-Colorimetric-Reader-with-Color-Subtraction-IEEE-Sensors-2019/blob/master/resources/red.jpg" height="220">
</p>

In the other two folders `blue` and `red` are the images of pH test strips with the solution color catcher (SCC) using blue and red dyed pH buffer solutions, respectively. They are used for the color subtration performance evaluation. 

The background images are also provided with each image. In the `tags.json` files under each folder stores the bounding boxes of the color areas in the images and the corresponding true pH values of the solutions measured using a lab-based pH meter.


## Validation

Run `python color_subtraction.py` to train and validate the color subtraction model proposed in the paper.


## Publication

The oral presentation in IEEE Sensors 2019:

https://www.youtube.com/watch?v=NfHZv8c8GZM


Please cite our paper if you find this work helpful:

https://ieeexplore.ieee.org/document/8956565

https://pure.qub.ac.uk/files/192853239/Zhao_Smartphone_Accepted.pdf


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
