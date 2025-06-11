"""
Script to visualize the effect of different time embeddings on the output of a Diffusion Transformer (DiT) model.

This script loads a pre-trained DiT model (VAE, Conditioner, Denoiser components) from a configuration
file and a checkpoint. It then generates a series of images using a fixed initial latent noise
and class label, but varies the time step ('t') provided to the denoiser. This allows for
observing how the model's output evolves or changes based on the time conditioning.

Inputs:
  --config_path: Path to the YAML configuration file for the model.
  --checkpoint_path: Path to the model checkpoint (.ckpt file).
  --output_dir: Directory where the generated images will be saved.
  --num_images: Number of images to generate per time step (batch size for fixed noise). Default is 1.
  --num_time_steps: Number of different time steps to visualize. Default is 10.
  --class_label: Integer class label for conditioning. Default is 0.
  --device: Device to run on ('cuda' or 'cpu'). Default is 'cuda'.

Outputs:
  A series of PNG images saved in the specified output directory. Each image is named
  according to the time step value and class label used for its generation, e.g.,
  `time_0.5000_label_0.png`. If `num_images` > 1, an image index is also added.

Example Usage:
  python tools/time_embedding_effect.py \
    --config_path configs/your_model_config.yaml \
    --checkpoint_path checkpoints/your_model.ckpt \
    --output_dir results/time_effect_experiment \
    --num_images 1 \
    --num_time_steps 20 \
    --class_label 5 \
    --device cuda
"""
import argparse
import os
import yaml
import torch
from omegaconf import OmegaConf
# from lightning.pytorch.cli import LightningCLI # For understanding, not direct use yet
from torchvision.utils import save_image
# import matplotlib.pyplot as plt # Not strictly needed if save_image works for all cases
import importlib

# Assuming these paths are correct relative to the project root
# If not, they might need adjustment when the script is run
# or the script needs to be run from the project root.
# Assuming these paths are correct relative to the project root
# If not, they might need adjustment when the script is run
# or the script needs to be run from the project root.
# No specific model imports like LightningModel are needed if instantiating from config directly.

def instantiate_from_config(config: OmegaConf):
    """
    Instantiates an object from an OmegaConf configuration object.
    The configuration object must have 'class_path' and 'init_args'.
    'class_path' specifies the module and class name (e.g., 'some_module.SomeClass').
    'init_args' is a dictionary of arguments to pass to the class constructor.
    """
    if not hasattr(config, "class_path") or not hasattr(config, "init_args"):
        raise ValueError("Configuration object must have 'class_path' and 'init_args' attributes.")

    try:
        module_path, class_name = config.class_path.rsplit(".", 1)
        module = importlib.import_module(module_path, package=None)
        class_ = getattr(module, class_name)
        return class_(**config.init_args)
    except ImportError as e:
        raise ImportError(f"Could not import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Could not find class {class_name} in module {module_path}: {e}")
    except Exception as e:
        raise Exception(f"Error instantiating {config.class_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the effect of time embeddings on generated images from a DiT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the YAML configuration file (e.g., configs/repa_improved_dit_large.yaml).")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint (.ckpt file).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the generated images.")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of images to generate per time step (effectively batch size for the fixed noise).")
    parser.add_argument("--num_time_steps", type=int, default=10,
                        help="Number of different time steps to visualize, spread between 0.05 and 0.95.")
    parser.add_argument("--class_label", type=int, default=0,
                        help="Integer class label for conditioning the generation.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run the model on ('cuda' or 'cpu').")

    args = parser.parse_args()

    print("INFO: Starting script...")

    # --- Argument Validation and Output Directory Setup ---
    if not os.path.isfile(args.config_path):
        print(f"ERROR: Configuration file not found at {args.config_path}")
        return
    if not os.path.isfile(args.checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {args.checkpoint_path}")
        return

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"INFO: Output directory set to: {args.output_dir}")
    except OSError as e:
        print(f"ERROR: Could not create output directory {args.output_dir}: {e}")
        return

    # --- Load Configuration ---
    print(f"INFO: Loading configuration from {args.config_path}...")
    try:
        conf = OmegaConf.load(args.config_path)
        print("INFO: Configuration loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load configuration file {args.config_path}: {e}")
        return

    # --- Device Setup ---
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA selected but not available. Switching to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"INFO: Using device: {device}")

    # --- Instantiate Model Components ---
    print("INFO: Instantiating model components (VAE, Conditioner, Denoiser)...")
    try:
        vae = instantiate_from_config(conf.model.vae)
        print("INFO: VAE instantiated.")
        conditioner = instantiate_from_config(conf.model.conditioner)
        print("INFO: Conditioner instantiated.")
        denoiser = instantiate_from_config(conf.model.denoiser)
        print("INFO: Denoiser instantiated.")
    except Exception as e:
        print(f"ERROR: Failed to instantiate one or more model components: {e}")
        return

    # --- Load Checkpoint ---
    print(f"INFO: Loading checkpoint from {args.checkpoint_path}...")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if 'state_dict' not in checkpoint:
            print("ERROR: 'state_dict' key not found in the checkpoint.")
            return
        full_state_dict = checkpoint['state_dict']
        print("INFO: Checkpoint loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return

    # Filter and load state_dicts for each component
    print("INFO: Filtering and loading state_dicts for model components...")
    vae_state_dict = {k.replace("vae.", ""): v for k, v in full_state_dict.items() if k.startswith("vae.")}
    conditioner_state_dict = {k.replace("conditioner.", ""): v for k, v in full_state_dict.items() if k.startswith("conditioner.")}
    denoiser_state_dict = {k.replace("denoiser.", ""): v for k, v in full_state_dict.items() if k.startswith("denoiser.")}

    missing_keys_report = {}
    unexpected_keys_report = {}

    try:
        m, u = vae.load_state_dict(vae_state_dict, strict=False); missing_keys_report['vae'] = m; unexpected_keys_report['vae'] = u
        print("INFO: VAE state_dict loaded.")
        m, u = conditioner.load_state_dict(conditioner_state_dict, strict=False); missing_keys_report['conditioner'] = m; unexpected_keys_report['conditioner'] = u
        print("INFO: Conditioner state_dict loaded.")
        m, u = denoiser.load_state_dict(denoiser_state_dict, strict=False); missing_keys_report['denoiser'] = m; unexpected_keys_report['denoiser'] = u
        print("INFO: Denoiser state_dict loaded.")

        for component, keys in missing_keys_report.items():
            if keys: print(f"WARNING: Missing keys for {component}: {keys}")
        for component, keys in unexpected_keys_report.items():
            if keys: print(f"WARNING: Unexpected keys for {component}: {keys}")

    except Exception as e:
        print(f"ERROR: Failed to load state_dicts into model components: {e}")
        return

    # Move models to device and set to evaluation mode
    vae.to(device).eval()
    conditioner.to(device).eval()
    denoiser.to(device).eval()
    print("INFO: All model components moved to device and set to evaluation mode.")

    # --- Prepare Inputs for Generation ---
    print("INFO: Preparing inputs for image generation...")
    try:
        # Fixed Latent Noise: The initial noise vector that will be 'denoised'.
        # Its dimensions should match what the denoiser expects as input.
        in_channels = conf.model.denoiser.init_args.in_channels # Number of channels in the latent space.

        # Determine latent dimensions (height, width)
        # Primary source: conf.data.latent_shape = [channels, height, width]
        if hasattr(conf.data, "latent_shape") and isinstance(conf.data.latent_shape, (list, OmegaConf.ListConfig)) and len(conf.data.latent_shape) == 3:
            latent_h = conf.data.latent_shape[1]
            latent_w = conf.data.latent_shape[2]
            if conf.data.latent_shape[0] != in_channels:
                 print(f"WARNING: conf.data.latent_shape[0] ({conf.data.latent_shape[0]}) "
                       f"differs from denoiser's in_channels ({in_channels}). Using denoiser's in_channels for noise generation.")
        # Fallback: Infer from VAE config (if available) and denoiser's image_size (assuming it's latent size for DiT)
        elif hasattr(conf.model.vae.init_args, "latent_channels") and hasattr(conf.model.denoiser.init_args, "image_size"):
            print("WARNING: 'conf.data.latent_shape' not found or invalid. Attempting to infer latent dimensions from VAE and Denoiser configs.")
            if conf.model.denoiser.init_args.in_channels != conf.model.vae.init_args.latent_channels:
                 print(f"WARNING: Denoiser in_channels ({conf.model.denoiser.init_args.in_channels}) "
                       f"differs from VAE latent_channels ({conf.model.vae.init_args.latent_channels}). Using denoiser's in_channels.")
            # Assuming denoiser.init_args.image_size refers to the spatial dimensions of the latent representation for DiT models.
            image_size = conf.model.denoiser.init_args.image_size
            latent_h, latent_w = image_size, image_size
        else:
            print("ERROR: Could not determine latent dimensions (height, width). "
                  "Please ensure 'conf.data.latent_shape' or relevant model config fields are present.")
            return

        x_fixed_latent = torch.randn(args.num_images, in_channels, latent_h, latent_w, device=device)
        print(f"INFO: Generated fixed latent noise 'x_fixed_latent' with shape: {x_fixed_latent.shape}")

        # Class Labels: Integer indices for class conditioning.
        y_indices = torch.tensor([args.class_label] * args.num_images, device=device, dtype=torch.long)
        print(f"INFO: Generated class labels 'y_indices' with shape: {y_indices.shape}, value: {y_indices[0].item() if args.num_images > 0 else 'N/A'}")

        # Time Steps: A sequence of scalar time values (normalized, e.g., 0 to 1).
        # These will be used to condition the denoiser at different stages.
        t_values = torch.linspace(0.05, 0.95, args.num_time_steps, device=device) # Avoids extreme values like 0 or 1 if they cause issues.
        print(f"INFO: Generated time steps 't_values' (linspace 0.05-0.95) with shape: {t_values.shape}")

    except AttributeError as e:
        print(f"ERROR: Configuration missing expected attribute for input generation: {e}")
        print("       Please ensure your config file has: "
              "conf.model.denoiser.init_args.in_channels, and ideally conf.data.latent_shape. "
              "Alternatively, check VAE/Denoiser model component configs for fallback latent dim inference.")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during input generation: {e}")
        return

    # --- Core Image Generation Loop ---
    print(f"\nINFO: Starting image generation loop for {args.num_time_steps} time steps...")
    with torch.no_grad(): # Disable gradient calculations for inference.
        for i, t_current_scalar in enumerate(t_values):
            current_t_value = t_current_scalar.item()
            print(f"STATUS: Processing time step {i+1}/{args.num_time_steps} (t_scalar={current_t_value:.4f})")

            # Prepare 't' for the Denoiser:
            # The denoiser (especially DiT) typically expects a batch of time values (one per item in the batch).
            # These should be float tensors.
            t_for_denoiser = torch.full(
                (args.num_images,),       # Shape: (batch_size,)
                current_t_value,          # Scalar time value for this step
                device=device,
                dtype=torch.float         # Ensure float type for time embedding modules
            )

            try:
                # 1. Denoiser Call:
                #    Input: fixed latent noise, current time step, class conditioning.
                #    Output: theoretically, a "less noisy" version of the latent.
                #    Note: The exact interaction with the 'conditioner' might vary.
                #          If 'conditioner' is a separate module that preprocesses 'y_indices'
                #          into embeddings, that step would happen before this call.
                #          Here, 'y_indices' is passed directly, assuming the 'denoiser' or an
                #          internal component handles the conditioning.
                denoiser_output = denoiser(x_fixed_latent, t_for_denoiser, y_indices)

                # 2. VAE Decoding:
                #    Convert the denoiser's output (from latent space) back to pixel space.
                decoded_images = vae.decode(denoiser_output)

                # 3. Normalize for Saving:
                #    VAE output is typically in [-1, 1] range. Normalize to [0, 1] for image saving.
                normalized_images = (decoded_images / 2 + 0.5).clamp(0, 1)

                # 4. Save Image(s):
                #    Handle batch saving if num_images > 1.
                if args.num_images > 1:
                    for j in range(args.num_images):
                        output_filename = os.path.join(
                            args.output_dir,
                            f"time_{current_t_value:.4f}_label_{args.class_label}_imgidx_{j}.png"
                        )
                        save_image(normalized_images[j:j+1], output_filename) # Save individual image from batch
                        # print(f"INFO: Saved image: {output_filename}") # Can be too verbose
                else:
                    output_filename = os.path.join(
                        args.output_dir,
                        f"time_{current_t_value:.4f}_label_{args.class_label}.png"
                    )
                    save_image(normalized_images, output_filename)
                # A single print after all images for this timestep are saved, or per image if preferred.
                print(f"INFO: Saved images for t={current_t_value:.4f} to {args.output_dir}")


            except Exception as e:
                print(f"ERROR: Failed during generation/saving for t={current_t_value:.4f}: {e}")
                # Decide if to 'continue' to the next time step or 'return'/'raise e'
                continue # Continue with the next time step

    print("\nINFO: Image generation loop finished.")
    print("INFO: Script finished.")

if __name__ == "__main__":
    main()
