# geometric_analysis
Experiments on geometric analysis of manifold transformations.




#### Loss gradient partial derivatives with respect to the Intermediate layer output  during adversarial attack optimization

<pre>
```
python diffae/loss_grads_wrt_intermediate_maps.py --desired_norm_l_inf 0.27 --which_gpu 1 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad --chosen_space_ind 14
```
</pre>

Observations: 
1. The shapes of loss gradients with respect to intermediate layer outputs is equivalent to shapes of intermediate layer outputs.
2. individual layer parameters do not have gradients during adversarial attack optimization.