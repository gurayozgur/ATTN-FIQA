import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.QualityModel import QualityModel

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Extract quality scores using VIT-FIQA models')

    parser.add_argument('--data-dir', type=str,
                        default='../data/',
                        help='Root directory for evaluation dataset')
    parser.add_argument('--output-dir', type=str,
                        default='../results/extracted_quality_scores',
                        help='Directory to save quality scores')
    parser.add_argument('--datasets', type=str,
                        default='lfw,calfw,cplfw,agedb_30,cfp_fp,XQLFW',
                        help='Comma-separated list of datasets (e.g., XQLFW,lfw,calfw,agedb_30,cfp_fp,cplfw,IJBC)')
    parser.add_argument('--gpu-id', type=int,
                        default=0,
                        help='GPU device ID')
    parser.add_argument('--model-path', type=str,
                        default="../pretrained/",
                        help='Path to directory containing pretrained model')
    parser.add_argument('--model-name', type=str,
                        default="minchul_cvlface_adaface_vit_base_webface4m.pt",
                        help='Name of the pretrained model file')
    parser.add_argument('--backbone', type=str,
                        default="vitb",
                        choices=['vits', 'vitb'],
                        help='Backbone architecture (vits, or vitb)')
    parser.add_argument('--batch-size', type=int,
                        default=32,
                        help='Batch size for processing')
    parser.add_argument('--color-channel', type=str,
                        default="BGR",
                        choices=['BGR', 'RGB'],
                        help='Input image color channel format')

    return parser.parse_args(argv)


def read_image_list(image_list_file, image_dir=''):
    """Read list of image paths from file."""
    image_lists = []
    with open(image_list_file) as f:
        absolute_list = f.readlines()
        for line in absolute_list:
            image_lists.append(os.path.join(image_dir, line.rstrip()))
    return image_lists, absolute_list


def main(param):
    datasets = param.datasets.split(',')

    # Parse blocks_to_use from command line
    blocks_to_use = None
    if param.blocks_to_use is not None:
        # Parse comma-separated block indices
        blocks_to_use = [int(x.strip()) for x in param.blocks_to_use.split(',')]
    elif param.num_blocks is not None:
        # Backward compatibility: convert num_blocks to blocks_to_use
        blocks_to_use = list(range(param.num_blocks))

    print(f"Initializing Quality Model...")
    print(f"  Model path: {param.model_path}")
    print(f"  Model name: {param.model_name}")
    print(f"  Backbone: {param.backbone}")
    print(f"  GPU ID: {param.gpu_id}")

    # Initialize face quality model
    face_model = QualityModel(
        model_path=param.model_path,
        model_name=param.model_name,
        gpu_id=param.gpu_id,
        backbone=param.backbone
    )

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")

        # Construct paths
        root = param.data_dir
        image_list_file = os.path.join(param.data_dir, 'quality_data', dataset_name, 'image_path_list.txt')

        if not os.path.exists(image_list_file):
            print(f"  Warning: Image list file not found: {image_list_file}")
            print(f"  Skipping dataset {dataset_name}")
            continue

        # Read image paths
        image_list, absolute_list = read_image_list(image_list_file, root)
        print(f"  Found {len(image_list)} images")

        # Extract features and quality scores
        print(f"  Extracting quality getQualityScore...")
        _, quality = face_model.get_batch_feature(
            image_list,
            batch_size=param.batch_size,
            color=param.color_channel
        )

        # Create output directory
        output_dataset_dir = os.path.join(param.output_dir, dataset_name)
        os.makedirs(output_dataset_dir, exist_ok=True)

        # Set backbone label based on model_name
        if param.backbone == 'vitb':
            if '12m' in param.model_name.lower():
                backbone_label = 'VITB12M'
            elif '4m' in param.model_name.lower():
                backbone_label = 'VITB4M'
            else:
                backbone_label = 'VITB'
        elif param.backbone == 'vits':
            backbone_label = 'VITS'
        else:
            backbone_label = param.backbone.upper()

        # Add face recognition method to label if present in model name
        model_name_lower = param.model_name.lower()
        if 'arcface' in model_name_lower:
            backbone_label += '_ARCFACE'
        elif 'adaface' in model_name_lower:
            backbone_label += '_ADAFACE'

        filename = f"ATTN-FIQA_{backbone_label}_{dataset_name}.txt"
        output_file = os.path.join(output_dataset_dir, filename)

        # Save quality scores
        with open(output_file, "w") as f:
            for i in range(len(quality)):
                f.write(f"{absolute_list[i].rstrip()} {quality[i][0]:.15f}\n")

        print(f"  Saved quality scores to: {output_file}")
        print(f"  Quality score statistics:")
        print(f"    Mean: {quality.mean():.15f}")
        print(f"    Std:  {quality.std():.15f}")
        print(f"    Min:  {quality.min():.15f}")
        print(f"    Max:  {quality.max():.15f}")


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
