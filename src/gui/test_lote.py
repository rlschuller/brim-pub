import os
from glob import glob

import nrrd
import numpy as np
import PySimpleGUI as sg
from tqdm.tk import tqdm

folder = sg.popup_get_folder("Esolha o lote a ser testado")

filepath = os.path.join(
    os.path.dirname(folder), os.path.basename(folder) + "-erros.txt"
)


exam_labels = ["A1", "A2", "A3", "A4", "A5", "A6"]

f = open(filepath, "w")

num_errors = 0

for subject_dir in tqdm(
    sorted(glob(f"{folder}/*")), desc="Verificando arquivos AX.nrrd"
):
    exams = []
    if len(glob(f"{subject_dir}/A*.nrrd")) > 6:
        print(f"{subject_dir}: mais de 6 arquivos do tipo 'A*.nrrd'.", file=f)
        num_errors += 1

    try:
        for label in exam_labels:
            data, header = nrrd.read(os.path.join(subject_dir, label + ".nrrd"))
            exams.append([data, header])

        is_4d = [len(e[0].shape) > 3 for e in exams]

        if any(is_4d):
            if sum(is_4d) > 1:
                print(f"{subject_dir}: mais de um arquivo com múltiplas camadas!")
                exit(1)
            else:
                i_4d = np.argmax(is_4d)

                exam_4d = exams[i_4d]

                correspondence_vector = ["na"] * exam_4d[0].shape[0]

                for i in range(len(exams)):
                    if i != i_4d:
                        for j in range(exam_4d[0].shape[0]):
                            if np.array_equal(exam_4d[0][j, :, :, :], exams[i][0]):
                                correspondence_vector[j] = exam_labels[i]
                print(
                    f"{subject_dir}: arquivo_de_múltiplas_camadas='{exam_labels[i_4d]}.nrrd',  correspondências={correspondence_vector}",
                    file=f,
                )
                num_errors += 1

    except FileNotFoundError as err:
        print(f"{subject_dir}: {err}", file=f)
        num_errors += 1

f.close()

if num_errors > 0:
    sg.popup(f"{num_errors} erros salvos em\n{filepath}")
else:
    sg.popup("Nenhum erro encontrado!")
