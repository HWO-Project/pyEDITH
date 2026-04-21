import os
import pandas as pd
import numpy as np
from astropy import units as u
from pyEDITH import calculate_texp, lambda_d_to_arcsec
from pyEDITH.units import *
import matplotlib.pyplot as plt
import hwostyle

hwostyle.use("light")

hwocolor = hwostyle.palette

# Revert hwostyle changes
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True
# Load HPIC
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_file_path = os.path.join(script_dir, "ETC_cal_detect.xlsx")
hpic = pd.read_csv(os.path.join(script_dir, "full_HPIC.txt"), sep="|")


def to_arcsec(quantity, observer_distance):
    return np.arctan(quantity / observer_distance).to_value(ARCSEC)


def fluxes_to_magnitudes(F_star, F_p):
    return -2.5 * np.log10(np.array(F_p) / np.array(F_star))


def process_star(name):
    hip_name = int(name.strip("HIP "))

    for wavelength in ["500nm", "1000nm"]:
        print(f"\nProcessing {name} at {wavelength}")

        if wavelength == "500nm":
            columns = "A,D,E,F"
        else:
            columns = "A,J,K,L"

        df = pd.read_excel(
            excel_file_path,
            sheet_name=" ".join([name, "(Detection)"]),
            skiprows=[1, 2, 3, 4, 5, 6],
            usecols=columns,
        )

        df.columns = ["parameter", "AYO", "EBS", "EXOSIMS"]

        # Prepare input parameters for pyEDITH calculation
        input_params = prepare_input_params(df, hpic, hip_name, "AYO")

        import logging

        yippy_logger = logging.getLogger("yippy")
        yippy_logger.setLevel(logging.ERROR)
        # Run pyEDITH calculation
        texp, validation_output = calculate_texp(input_params, ETC_validation=True)

        # # Compare with Ayo's results
        # compare_with_ayo(
        #     name, wavelength, validation_output[0], df[["parameter", "AYO"]]
        # )
        compare_all_codes(name, wavelength, validation_output[0], df)

        print(f"Exposure time: {texp}")


def prepare_input_params(df, hpic, hip_name, code):

    input = {
        "F0": np.array([df.loc[df["parameter"] == "F_0", code].iloc[0]]),
        "mag": np.array([df.loc[df["parameter"] == "m_lambda", code].iloc[0]]),
        "Lstar": 10 ** float(hpic[hpic.hip_name == hip_name].st_lum.iloc[0]),
        "magV": float(hpic[hpic.hip_name == hip_name].sy_vmag.iloc[0]),
        "stellar_radius": (float(hpic[hpic.hip_name == hip_name].st_rad.iloc[0])),
        "distance": float(hpic[hpic.hip_name == hip_name].sy_dist.iloc[0]),
        "diameter": df.loc[df["parameter"] == "D", code].iloc[0],
        "unobscured_area": (1.0 - 0.121),
        "photometric_aperture_radius": 0.85,
        "Tcore": df.loc[df["parameter"] == "T_core", code].iloc[0] * DIMENSIONLESS,
        # "psf_trunc_ratio": df.loc[df["parameter"] == "psf_trunc_ratio", code].iloc[0],
        "det_npix_input": np.float64(
            df.loc[df["parameter"] == "det_npix", code].iloc[0]
        ),
        "wavelength": np.array(
            [df.loc[df["parameter"] == "λ", code].iloc[0] / 1000]
        ),  # nm to micron
        "bandwidth": df.loc[df["parameter"] == "Δλ", code].iloc[0]
        / df.loc[df["parameter"] == "λ", code].iloc[0],
        "nzodis": df.loc[df["parameter"] == "nzodis", code].iloc[0],
        "snr": np.array([df.loc[df["parameter"] == "SNR", code].iloc[0]]),
        "toverhead_fixed": df.loc[df["parameter"] == "t_overhead,static", code].iloc[0],
        "toverhead_multi": df.loc[df["parameter"] == "t_overhead,dynamic", code].iloc[
            0
        ],
        "DC": np.array([df.loc[df["parameter"] == "det_DC", code].iloc[0]]),
        "RN": np.array([df.loc[df["parameter"] == "det_RN", code].iloc[0]]),
        "CIC": np.array([df.loc[df["parameter"] == "det_CIC", code].iloc[0]]),
        "dQE": np.array([df.loc[df["parameter"] == "dQE", code].iloc[0]]),
        "QE": np.array([df.loc[df["parameter"] == "QE", code].iloc[0]]),
        "T_optical": np.array([df.loc[df["parameter"] == "T_optical", code].iloc[0]]),
        "ra": float(hpic[hpic.hip_name == hip_name].ra.iloc[0]),
        "dec": float(hpic[hpic.hip_name == hip_name].dec.iloc[0]),
        "separation": lambda_d_to_arcsec(
            value_lod=df.loc[df["parameter"] == "sp", code].iloc[0],
            wavelength=np.array(df.loc[df["parameter"] == "λ", code].iloc[0] / 1000)
            * u.micron,
            diameter=df.loc[df["parameter"] == "D", code].iloc[0] * u.m,
        ).value,
        "CRb_multiplier": 2.0,
        "Fstar": np.array([df.loc[df["parameter"] == "F_star", code].iloc[0]]),
        "Fp": np.array([df.loc[df["parameter"] == "F_p", code].iloc[0]]),
        "observatory_preset": "EAC1",
        "observing_mode": "IMAGER",
        "delta_mag_min": 25,
        "nchannels": 1,
        "epswarmTrcold": [0],
        "t_photon_count_input": np.float64(
            df.loc[df["parameter"] == "t_photon_count", code].iloc[0]
        ),
        "az_avg": True,
        "noisefloor_PPF": 1 / 0.029,
    }

    input["delta_mag"] = fluxes_to_magnitudes(input["Fstar"], input["Fp"])

    return input


# def get_expected_output(df, code):
#     return {
#         "F0": np.array(df.loc[df["parameter"] == "F_0", code].iloc[0]),
#         "magstar": np.array(df.loc[df["parameter"] == "m_lambda", code].iloc[0]),
#         "Lstar": np.float64(df.loc[df["parameter"] == "L_star", code].iloc[0]),
#         "dist": np.float64(df.loc[df["parameter"] == "dist", code].iloc[0]),
#         "D": np.float64(df.loc[df["parameter"] == "D", code].iloc[0]),
#         "A_cm": np.float64(df.loc[df["parameter"] == "A", code].iloc[0]),
#         "wavelength": np.float64(df.loc[df["parameter"] == "λ", code].iloc[0]),
#         "deltalambda_nm": np.float64(df.loc[df["parameter"] == "Δλ", code].iloc[0]),
#         "snr": np.float64(df.loc[df["parameter"] == "SNR", code].iloc[0]),
#         "nzodis": np.float64(df.loc[df["parameter"] == "nzodis", code].iloc[0]),
#         "toverhead_fixed": np.float64(
#             df.loc[df["parameter"] == "t_overhead,static", code].iloc[0]
#         ),
#         "toverhead_multi": np.float64(
#             df.loc[df["parameter"] == "t_overhead,dynamic", code].iloc[0]
#         ),
#         "det_DC": np.float64(df.loc[df["parameter"] == "det_DC", code].iloc[0]),
#         "det_RN": np.float64(df.loc[df["parameter"] == "det_RN", code].iloc[0]),
#         "det_CIC": np.float64(df.loc[df["parameter"] == "det_CIC", code].iloc[0]),
#         "det_tread": np.float64(df.loc[df["parameter"] == "det_tread", code].iloc[0]),
#         "det_pixscale_mas": np.float64(
#             df.loc[df["parameter"] == "det_pixscale", code].iloc[0]
#         ),
#         "dQE": np.float64(df.loc[df["parameter"] == "dQE", code].iloc[0]),
#         "QE": np.float64(df.loc[df["parameter"] == "QE", code].iloc[0]),
#         "T_optical": np.float64(df.loc[df["parameter"] == "T_optical", code].iloc[0]),
#         "Fs_over_F0": np.float64(df.loc[df["parameter"] == "F_star", code].iloc[0]),
#         "Fp": np.float64(df.loc[df["parameter"] == "F_p", code].iloc[0]),
#         "Fzodi": np.float64(df.loc[df["parameter"] == "F_zodi", code].iloc[0]),
#         "Fexozodi": np.array(df.loc[df["parameter"] == "F_exozodi", code]),
#         "sp_lod": np.array(df.loc[df["parameter"] == "sp", code]),
#         "omega_lod": np.float64(df.loc[df["parameter"] == "Ω_core", code].iloc[0]),
#         "T_core or photometric_aperture_throughput": np.float64(
#             df.loc[df["parameter"] == "T_core", code].iloc[0]
#         ),
#         "Istar*oneopixscale2 in (l/D)^-2": np.float64(
#             df.loc[df["parameter"] == "I_star", code].iloc[0]
#         ),
#         # "contrast * offset PSF peak *oneopixscale2  in (l/D)^-2 (unused)": np.float64(
#         #     3.9e-14
#         # ),
#         "skytrans*oneopixscale2  in (l/D)^-2": np.float64(
#             df.loc[df["parameter"] == "skytrans", code].iloc[0]
#         ),
#         "det_npix": np.float64(df.loc[df["parameter"] == "det_npix", code].iloc[0]),
#         # "t_photon_count_ETCVALIDATION": np.float64(
#         #     df.loc[df["parameter"] == "t_photon_count", code].iloc[0]
#         # ),
#         "t_photon_count": np.float64(
#             df.loc[df["parameter"] == "t_photon_count", code].iloc[0]
#         ),
#         "CRp": np.float64(df.loc[df["parameter"] == "CR_p", code].iloc[0]),
#         "CRbs": np.float64(df.loc[df["parameter"] == "CR_bs", code].iloc[0]),
#         "CRbz": np.float64(df.loc[df["parameter"] == "CR_bz", code].iloc[0]),
#         "CRbez": np.array(df.loc[df["parameter"] == "CR_bez", code]),
#         "CRbbin": np.float64(df.loc[df["parameter"] == "CR_bstray", code].iloc[0]),
#         "CRbd": np.float64(df.loc[df["parameter"] == "CR_bd", code].iloc[0]),
#         "CRnf": np.float64(df.loc[df["parameter"] == "CR_NF", code].iloc[0]),
#         "sciencetime": np.float64(df.loc[df["parameter"] == "t_science", code].iloc[0]),
#         "exptime": np.float64(df.loc[df["parameter"] == "t_exp", code].iloc[0]),
#     }


# def compare_with_ayo(name, lamb, pyedith_output, df):
#     print(f"Comparing with Ayo's results for {name} at {lamb}")
#     expected_output = get_expected_output(df, "AYO")
#     errors = []

#     for key, expected_value in expected_output.items():
#         calculated_value = pyedith_output[key]
#         if hasattr(calculated_value, "value"):
#             calculated_value = calculated_value.value

#         try:
#             np.testing.assert_allclose(
#                 calculated_value,
#                 expected_value,
#                 rtol=1e-1,
#                 err_msg=f"Mismatch in {key} for test case: {name}",
#             )
#         except AssertionError as e:
#             errors.append(
#                 f"-- {key}: FAILED - Expected: {expected_value}, Calculated: {calculated_value}"
#             )

#     if len(errors) == 0:
#         print(f"Test case '{name}' at {lamb} passed successfully!")
#     else:
#         print(f"Test case '{name}' at {lamb} had some errors: \n" + "\n".join(errors))


def compare_all_codes(name, wavelength, pyedith_output, df):
    print(f"Comparing with all codes for {name} at {wavelength}")

    key_translation = {
        "Fs_over_F0": "F_star",
        "Fp": "F_p",
        "Fzodi": "F_zodi",
        "Fexozodi": "F_exozodi",
        "T_core or photometric_aperture_throughput": "T_core",
        "omega_lod": "omega_core",
        "Istar*oneopixscale2 in (l/D)^-2": "I_star",
        "skytrans*oneopixscale2  in (l/D)^-2": "skytrans",
        "det_npix": "det_npix",
        "t_photon_count": "t_photon_count",
        "CRp": "CR_p",
        "CRbs": "CR_bs",
        "CRbez": "CR_bez",
        "CRbz": "CR_bz",
        "CRbd": "CR_bd",
        "CRnf": "CR_NF",
        "sciencetime": "t_science",
        "exptime": "t_exp",
    }
    renamed_pyedith_output = {}

    for old_key, value in pyedith_output.items():
        if old_key in key_translation:
            new_key = key_translation[old_key]
            renamed_pyedith_output[new_key] = value

    comparisons = {}
    for key in renamed_pyedith_output.keys():
        if key in df["parameter"].values:
            other_results = {
                code: df.loc[df["parameter"] == key, code].iloc[0]
                for code in ["AYO", "EBS", "EXOSIMS"]
            }
            if key == "CR_NF":
                print("multiplying CR_NF by SNR again for the validation")

                for code in other_results.keys():
                    other_results[code] *= 7
            comparisons[key] = compare_results(
                renamed_pyedith_output[key], other_results
            )

    visualize_comparisons(comparisons, name, wavelength)


def compare_results(pyedith_result, other_results):
    if hasattr(pyedith_result, "value"):
        pyedith_result = pyedith_result.value

    # calculate mean and std of the results from the other codes
    other_values = []
    for code in ["AYO", "EBS", "EXOSIMS"]:
        value = float(other_results[code])
        if not np.isnan(value):
            other_values.append(value)
        else:
            other_values.append(np.nan)

    mean = np.mean(other_values)
    std = np.std(other_values)

    return {
        "values": other_results,
        "pyedith": pyedith_result,
        "mean": mean,
        "std": std,
    }


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter


def visualize_comparisons(comparisons, name, wavelength):
    filename = name + "_" + str(wavelength)
    n_cols = 6
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), sharey=True)
    # fig.suptitle(f"Comparisons for {name} at {wavelength}", fontsize=20, y=1.02)

    axes = axes.flatten()
    codes = ["EXOSIMS", "EBS", "AYO", "pyEDITH"]
    colors = [hwocolor.cyan, hwocolor.pink, hwocolor.yellow, hwocolor.green]
    markers = [".", ".", ".", "*"]
    for i, (key, comparison) in enumerate(comparisons.items()):
        ax = axes[i]
        x = []
        y = []

        for j, code in enumerate(codes):
            value = (
                comparison["pyedith"]
                if code == "pyEDITH"
                else comparison["values"].get(code)
            )

            try:
                value = float(value)
                if not np.isnan(value):
                    x.append(value)
                    y.append(j)
                    ax.scatter(
                        value,
                        j,
                        c=colors[j],
                        marker=markers[j],
                        s=160,
                    )

                else:
                    raise ValueError
            except (ValueError, TypeError):
                ax.text(
                    0.5,
                    j,
                    "X",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=colors[j],
                )

        if not np.isnan(comparison["mean"]):
            ax.axvline(
                comparison["mean"],
                color="black",
                linestyle="--",
                linewidth=1,
                label="Mean",
            )
            ax.axvspan(
                comparison["mean"] - comparison["std"],
                comparison["mean"] + comparison["std"],
                alpha=0.2,
                color="gray",
                label="Std Dev",
            )
        x_min, x_max = min(x), max(x)
        x_range = x_max - x_min

        # log scale if it spans more than 2 orders of magnitude
        if len(set(x)) > 1 and max(x) / min(x) > 100:
            ax.set_xscale("log")
            ax.set_xlim(x_min / 1.1, x_max * 1.1)
        else:
            x_padding = x_range
            ax.set_xlim(max(0, x_min - x_padding), x_max + x_padding)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.offsetText.set_fontsize(9)
        ax.ticklabel_format(style="scientific", axis="x", scilimits=(-2, 2))

        ax.set_xlabel(key, fontsize=12)
        ax.set_yticks(range(4))
        ax.set_yticklabels(codes, rotation=45, fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.invert_yaxis()

        # Only show legend for the first subplot
        # if i == 0:
        #     ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))

    # Remove any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.5, wspace=0.15)
    plt.savefig(os.path.join(script_dir, filename + ".png"))


names = ["HIP 26779", "HIP 32439", "HIP 77052", "HIP 79672", "HIP 113283"]
for name in names:
    print("NAME", name)
    process_star(name)
