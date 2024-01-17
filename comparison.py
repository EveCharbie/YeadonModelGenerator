import numpy as np
import matplotlib.pyplot as plt
import biorbd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Comparison")
    parser.add_argument("measured_model_path", type=str, help="Path to the measured model biomod")
    parser.add_argument("generated_model_path", type=str, help="Path to the generated model biomod")
    args = parser.parse_args()
    com_comparison(args.measured_model_path, args.generated_model_path)


def com_comparison(measured_model, generated_model):
    model_gen = biorbd.Model(generated_model)
    model_ref = biorbd.Model(measured_model)
    model_gen_com_segment = model_gen.CoMbySegment(np.zeros((model_gen.nbQ(),)))
    model_ref_com_segment = model_ref.CoMbySegment(np.zeros((model_ref.nbQ(),)))
    model_gen_com_segment_list = []
    model_ref_com_segment_list = []

    for i in range(len(model_gen_com_segment)):
        model_gen_com_segment_list.append(model_gen_com_segment[i].to_array()[2])
        model_ref_com_segment_list.append(model_ref_com_segment[i].to_array()[2])

    plt.bar(range(len(model_gen_com_segment)), model_gen_com_segment_list, color="blue", alpha=0.7,
            label='Generated CoM')
    plt.bar(range(len(model_ref_com_segment)), model_ref_com_segment_list, color="green", alpha=0.7,
            label='Measured CoM')

    plt.xlabel('Index')
    plt.ylabel('CoM')
    plt.title('Comparison of CoM by segment between the generated model and the reference model')
    plt.legend()
    plt.savefig(f"{measured_model.split('/')[-1].split('_')[0]}_com_comparison.png")


if __name__ == "__main__":
    main()
