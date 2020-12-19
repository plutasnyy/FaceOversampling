from FaceOversampler import FaceOversampler
from psp_utils.common import tensor2im

oversampler = FaceOversampler("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17", ".", "pSp_weights/psp_ffhq_encode.pt")
oversampler.fit("oversampling/example_images")
tensor2im(oversampler.transform()[0]).save('org.jpg')
