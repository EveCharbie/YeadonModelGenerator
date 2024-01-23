import numpy as np
import matplotlib.pyplot as plt
import biorbd
import argparse


def com_comparison(measured_model, generated_model):
    model_gen = biorbd.Model(generated_model)
    model_ref = biorbd.Model(measured_model)
    model_gen_com_segment = model_gen.CoMbySegment(np.zeros((model_gen.nbQ(),)))
    model_ref_com_segment = model_ref.CoMbySegment(np.zeros((model_ref.nbQ(),)))
    model_gen_com_segment_list = []
    model_ref_com_segment_list = []
    diff_list = []
    for i in range(len(model_gen_com_segment)):
        model_gen_com_segment_list.append(model_gen_com_segment[i].to_array()[2])
        model_ref_com_segment_list.append(model_ref_com_segment[i].to_array()[2])
#        diff_list.append(model_ref_com_segment_list[i] - model_gen_com_segment_list[i])

    plt.bar(range(len(model_gen_com_segment)), model_gen_com_segment_list, color="blue", alpha=0.7,
            label='Generated CoM')
    plt.bar(range(len(model_ref_com_segment)), model_ref_com_segment_list, color="green", alpha=0.7,
            label='Measured CoM')
    plt.xlabel('Index')
    plt.ylabel('CoM')
    plt.title('Comparison of CoM by segment between the generated model and the reference model')
    plt.legend()
    plt.savefig(f"{measured_model.split('/')[-1].split('_')[0]}_com_comparison.png")
    plt.clf()


# TODO use the entire body Inertia matrix (right now only using the diagonals)
def inertia_comparison(measured_model, generated_model):
    model_gen = biorbd.Model(generated_model)
    model_ref = biorbd.Model(measured_model)
    model_gen_inertia_diagonal = np.diagonal(model_gen.bodyInertia(np.zeros((model_gen.nbQ(),))).to_array())
    model_ref_inertia_diagonal = np.diagonal(model_ref.bodyInertia(np.zeros((model_gen.nbQ(),))).to_array())
    model_gen_inertia_diag_component = []
    model_ref_inertia_diag_component = []
    for i in range(len(model_gen_inertia_diagonal)):
        model_gen_inertia_diag_component.append(model_gen_inertia_diagonal[i])
        model_ref_inertia_diag_component.append(model_ref_inertia_diagonal[i])
    plt.bar(range(len(model_gen_inertia_diagonal)), model_gen_inertia_diag_component, color="blue", alpha=0.7,
            label='Generated Inertia')
    plt.bar(range(len(model_ref_inertia_diagonal)), model_ref_inertia_diag_component, color="green", alpha=0.7,
            label='Measured Inertia')
    plt.xlabel('Index')
    plt.ylabel('Inertia')
    plt.title('Comparison of body Inertia between the generated model and the reference model')
    plt.legend()
    plt.savefig(f"{measured_model.split('/')[-1].split('_')[0]}_inertia_comparison.png")
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Comparison")
    parser.add_argument("measured_model_path", type=str, help="Path to the measured model biomod")
    parser.add_argument("generated_model_path", type=str, help="Path to the generated model biomod")
    args = parser.parse_args()
    inertia_comparison(args.measured_model_path, args.generated_model_path)

    com_comparison(args.measured_model_path, args.generated_model_path)


if __name__ == "__main__":
    main()
