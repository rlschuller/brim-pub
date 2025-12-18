import argparse

# import matplotlib.pyplot as plt
#import medpy.io
import napari
import nrrd

# import os


parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    "-p",
    default=["data/raw/01/A1.nrrd"],
    nargs="+",
    help="path to the nrrd files",
)

args = parser.parse_args()


viewer = napari.Viewer()


for path in args.path:
    print(path)
    data, header = nrrd.read(path)
    print("here")
    #print(data.shape)
    if len(data.shape) == 4:
        data = data[:, :, :, 0]

    #print(header.offset)
    volume_layer = viewer.add_image(data, name=path)

    # @volume_layer.mouse_drag_callbacks.append
    # def onclick(layer, event):
    #    print(event.position)
    #    i = round(event.position[0])
    #    j = round(event.position[1])
    #    k = round(event.position[2])
    #    if i > 0 and j > 0 and k > 0:
    #        vv = [
    #            [
    #                a.data[i, j, k]
    #                for a in viewer.layers
    #                if i < a.data.shape[0]
    #                and j < a.data.shape[1]
    #                and k < a.data.shape[2]
    #                and os.path.basename(str(a))
    #                in ["A1.nrrd", "A2.nrrd", "A3.nrrd", "A4.nrrd", "A5.nrrd"]
    #            ]
    #        ]
    #        print(
    #            [
    #                str(a)
    #                for a in viewer.layers
    #                if i < a.data.shape[0]
    #                and j < a.data.shape[1]
    #                and k < a.data.shape[2]
    #            ]
    #        )

    #    # format: <<ID:default_value>>
    #    labels = None
    #    h = None
    #    w = None

    #    cm = 1 / 2.54  # centimeters in inches
    #    if w is not None:
    #        plt.figure(figsize=(w * cm, h * cm))
    #    plt.rc("font", size=6)  # size=8 works better with LaTeX

    #    plt.clf()
    #    for ii in range(len(vv)):
    #        v = vv[ii]
    #        label = ""
    #        if labels and ii < len(labels):
    #            label = labels[ii]
    #            plt.plot(v, label=label)
    #        else:
    #            plt.plot(v)

    #    if labels:
    #        plt.legend(labels)

    #    plt.draw()
    #    plt.show()


napari.run()
