# Re-isotropic Segmentation for Subcellular Ultrastructure in Anisotropic EM Images

ReIsoSeg is a novel 3D segmentation framework aimed at overcoming the anisotropic problem in the segmentation of subcellular ultrastructure in electron microscopy (EM). Its main approach is to make it isotropic by achieving continuous consistency along different axes of features, thereby generating better features and enhancing the segmentation performance of the model.

<br />


For more information about ReIsoSeg, please read the following paper （Accepted by TMI 2024）: 
<br />

@ARTICLE{guo2024re,<br />
  author={Guo, Jinyue and Wang, Zejin and Zhai, Hao and Zhang, Yanchao and Liu, Jing and Han, Hua},<br />
  journal={IEEE Transactions on Medical Imaging}, <br />
  title={Re-isotropic Segmentation for Subcellular Ultrastructure in Anisotropic EM Images}, <br />
  year={2024},<br />
  volume={},<br />
  number={},<br />
  pages={1-1},<br />
  keywords={Anisotropic;Decoding;Anisotropic magnetoresistance;Convolution;Three-dimensional displays;Image resolution;Transformers;<br />Optimization;Image restoration;Synapses;Anisotropy;deep learning;electron microscopy;re-isotopic loss;volume segmentation},<br />
  doi={10.1109/TMI.2024.3511599}<br />
  }

## The basic idea of ReIsoSeg

<img src="https://github.com/fenglingbai/ReIsoSeg/blob/main/fig/p0_pipline.png" width="600px"> 

Overview of the proposed ReIsoSeg: it enforces the feature re-isotropy through an implicit self-supervised manner, increasing the performance of the encoder and decoder for superior segmentation.

## The architecture composition of ReIsoSeg

<img src="https://github.com/fenglingbai/ReIsoSeg/blob/main/fig/p2_ReIsoSeg.png" width="800px"> 

ReIsoSeg architecture: the PI Net and a series of feature transformations (e.g., axial permutation) constitute the PI module in the isotropic branch, and they enforce feature re-isotropy through self-supervision, which increases the segmentation performance of the encoder-decoder.

## Visualization of 3D Segmentation Results

<img src="https://github.com/fenglingbai/ReIsoSeg/blob/main/fig/p6_results_3d3.png" width="800px"> 

Partial results of suborganelle volume segmentation.

## How to use.