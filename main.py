from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import re
import time
from functools import wraps
import shapely
import os


def read_data_file(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        raw_file = f.readlines()

    list_dados = [line.split() for line in raw_file]
    float_raw_lines = [list(map(float, raw_line)) for raw_line in list_dados]
    return pd.DataFrame(float_raw_lines, columns=['lat', 'long', 'data_value'])


def read_contour_file(file_path: str) -> pd.DataFrame:
    line_split_comp = re.compile(r'\s*,')

    with open(file_path, 'r') as f:
        raw_file = f.readlines()

    l_raw_lines = [line_split_comp.split(raw_file_line.strip()) for raw_file_line in raw_file]
    l_raw_lines = list(filter(lambda item: bool(item[0]), l_raw_lines))
    float_raw_lines = [list(map(float, raw_line))[:2] for raw_line in l_raw_lines]
    header_line = float_raw_lines.pop(0)
    assert len(float_raw_lines) == int(header_line[0])
    return pd.DataFrame(float_raw_lines, columns=['lat', 'long'])


def apply_contour(contour_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    contour  = shapely.Polygon(contour_df.values)
    return data_df[data_df.apply(
        lambda point: contour.contains(shapely.Point(point["lat"], point["long"])),
        axis=1
    )]

def generate_and_save_image(interpolated_values: np.ndarray, polygon_vertices: np.ndarray, title_date: str, title_date_parsed: str, accumulated: bool = False) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(interpolated_values, extent=(min(polygon_vertices[:, 0]), max(polygon_vertices[:, 0]),
                                        min(polygon_vertices[:, 1]), max(polygon_vertices[:, 1])),
            origin='lower')

    plt.plot(polygon_vertices[:, 0], polygon_vertices[:, 1], 'r-')
    plt.colorbar(label='Previsão de chuva')
    plt.title(f"""Carmagros - Bacia do Grande - Previsão de chuva - {title_date_parsed if not accumulated else "Acumulada"}""")

    if not os.path.exists(os.path.join(os.getcwd(), "graphs")):
        os.mkdir(os.path.join(os.getcwd(), "graphs"))
    plt.savefig(os.path.join(os.getcwd(), "graphs", f"""graph-{title_date if not accumulated else "accumulated"}.png"""),bbox_inches='tight', dpi=300, facecolor="#FFF", edgecolor='none')

def get_interpolated_polygon(data_points: np.ndarray, polygon_vertices: np.ndarray) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(np.linspace(min(polygon_vertices[:, 0]), max(polygon_vertices[:, 0]), num=1000),
        np.linspace(min(polygon_vertices[:, 1]), max(polygon_vertices[:, 1]), num=1000))

    data_coords = data_points[:, :2]
    data_values = data_points[:, 2]

    return griddata(data_coords, data_values, (x_grid, y_grid), method='linear')


def main() -> None:
    contour_df: pd.DataFrame = read_contour_file('PSATCMG_CAMARGOS.bln')

    acc_df = pd.DataFrame({'lat': [], 'long': [], 'data_value': []})
    for index, file in enumerate(os.listdir(os.path.join(os.getcwd(), "forecast_files"))):
        data_df: pd.DataFrame = read_data_file(os.path.join(os.getcwd(), "forecast_files", file))
        contour_applied: pd.DataFrame = apply_contour(contour_df=contour_df, data_df=data_df)

        title_date = datetime.datetime.strptime(file[len("ETA40_p011221a"):-len(".dat")], "%d%m%y")
        title_date_parsed = title_date.strftime("%d/%m/%y")

        generate_and_save_image(
            get_interpolated_polygon(contour_applied.to_numpy(), contour_df.to_numpy()),
            contour_df.to_numpy(),
            title_date,
            title_date_parsed
        )

        acc_df['lat'] = contour_applied['lat']
        acc_df['long'] = contour_applied['long']
        if index == 0:
            acc_df['data_value'] = contour_applied['data_value']
        else:
            acc_df['data_value'] += contour_applied['data_value']

    contour_applied: pd.DataFrame = apply_contour(contour_df=contour_df, data_df=acc_df)

    generate_and_save_image(
        get_interpolated_polygon(contour_applied.to_numpy(), contour_df.to_numpy()),
        contour_df.to_numpy(),
        "",
        "",
        True
    )

if __name__ == '__main__':
    main()