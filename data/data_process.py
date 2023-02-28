from glob import glob
from os.path import join

from tqdm import tqdm
from cv2 import imread, imwrite

for n, file in enumerate(tqdm(
        glob(join("cartoon_A2B", "*", "*.png")),
        "Processing dataset", unit="files"
)):
    image, name = imread(file), "{}.png".format(n)
    person, cartoon = image[:, :256, :], image[:, 256:, :]
    imwrite(join("person", name), person)
    imwrite(join("cartoon", name), cartoon)
