#imports
import argparse
import nrrd
import numpy as np
import pathlib
#receve argumentos

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    type=str
)

parser.add_argument(
    "-o",
    default= False
)

parser.add_argument(
    "-sz",
    default=16,
    type=int
)

parser.add_argument(
    "-th",
    default= 0,
    type = int
)

args = parser.parse_args()


#Definindo paramentros

input_folder = args.i
output_folder = args.o
size = args.sz
threshold = args.th * 0.01 * (size**3)


#criando função de segmentação cúbica

def cube_seg(input_file, output_folder):
    print(input_file)
    seg, meta = nrrd.read(input_file)
    cube_seg = np.zeros(shape = seg.shape)

    if not output_folder:
        output_file = str(input_file.parent.parent.parent) + '/cube_seg/' + str(input_file.parent.name) + '/cube_seg.nrrd'
    
    new_path = pathlib.Path(str(input_file.parent.parent.parent) + '/cube_seg/' + str(input_file.parent.name))
    new_path.mkdir(exist_ok=True)

    shape = seg.shape

    for i in range(0, shape[0], size):
        for j in range(0, shape[1], size):
            for k in range(0, shape[2], size):
                
                cube = seg[i:i+size, j:j+size, k:k+size]
                
                if cube.sum() > threshold:
                    cube_seg[i:i+size, j:j+size, k:k+size] += 1
  

    nrrd.write(output_file, cube_seg, meta)

file_paths = pathlib.Path(input_folder).glob('*/*')
for f in file_paths:
    cube_seg(f, False)