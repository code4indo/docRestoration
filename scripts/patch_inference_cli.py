#!/usr/bin/env python3
"""
Patch-Based GAN-HTR Inference CLI
Command-line interface for universal document enhancement
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.patch_based_inference import UniversalGANHTRProcessor
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description='Universal GAN-HTR Document Enhancement with Patch-Based Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python %(prog)s --input document.png --output enhanced.png

  # Process directory
  python %(prog)s --input-dir documents/ --output-dir enhanced/

  # Custom configuration
  python %(prog)s --input document.png --patch-size 512 64 --overlap 0.3 --strategy sliding

  # High quality processing
  python %(prog)s --input document.png --blend-method gaussian --batch-size 2
        """
    )

    # Input/Output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                            help='Input image path')
    input_group.add_argument('--input-dir', '-I', type=str,
                            help='Input directory containing images')

    parser.add_argument('--output', '-o', type=str,
                       help='Output image path (for single image)')
    parser.add_argument('--output-dir', '-O', type=str,
                       help='Output directory (for batch processing)')

    # Model configuration
    parser.add_argument('--model-path', type=str,
                       default='dual_modal_gan/checkpoints/full_training_production_v1/best_model',
                       help='Path to model checkpoint directory')
    parser.add_argument('--checkpoint-name', type=str,
                       help='Specific checkpoint name (without .index)')

    # Patch configuration
    parser.add_argument('--patch-size', type=int, nargs=2, default=[512, 64],
                       help='Patch size (height width). Default: 512 64')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap ratio between patches (0.0-0.9). Default: 0.25')
    parser.add_argument('--strategy', choices=['sliding', 'grid', 'adaptive'],
                       default='sliding',
                       help='Patch extraction strategy. Default: sliding')

    # Processing configuration
    parser.add_argument('--blend-method', choices=['linear', 'gaussian'],
                       default='linear',
                       help='Blending method for reconstruction. Default: linear')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for processing. Default: 4')

    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--save-config', action='store_true',
                       help='Save configuration to JSON file')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode with simulated enhancement (no model required)')

    args = parser.parse_args()

    # Validate arguments
    if args.input and not args.output and not args.output_dir:
        # Auto-generate output path
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}")

    if args.input_dir and not args.output_dir:
        # Auto-generate output directory
        args.output_dir = str(Path(args.input_dir) / "enhanced")

    # Display configuration
    print("🚀 Patch-Based GAN-HTR Document Enhancement")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.verbose or args.save_config:
        print("📋 Configuration:")
        print(f"   Input: {args.input or args.input_dir}")
        print(f"   Output: {args.output or args.output_dir}")
        print(f"   Model: {args.model_path}")
        if args.checkpoint_name:
            print(f"   Checkpoint: {args.checkpoint_name}")
        print(f"   Patch size: {tuple(args.patch_size)}")
        print(f"   Overlap: {args.overlap:.1%}")
        print(f"   Strategy: {args.strategy}")
        print(f"   Blending: {args.blend_method}")
        print(f"   Batch size: {args.batch_size}")
        print()

    # Save configuration if requested
    if args.save_config:
        config = {
            'timestamp': datetime.now().isoformat(),
            'input': args.input or args.input_dir,
            'output': args.output or args.output_dir,
            'model_path': args.model_path,
            'checkpoint_name': args.checkpoint_name,
            'patch_size': tuple(args.patch_size),
            'overlap': args.overlap,
            'strategy': args.strategy,
            'blend_method': args.blend_method,
            'batch_size': args.batch_size
        }

        config_path = Path(args.output or args.output_dir) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Configuration saved to: {config_path}")
        print()

    # Demo mode
    if args.demo:
        print("🎨 Running in DEMO mode (simulated enhancement)")
        from scripts.test_patch_demo import process_with_simulation

        if args.input:
            result = process_with_simulation(
                args.input,
                Path(args.output).parent,
                patch_size=tuple(args.patch_size),
                overlap=args.overlap
            )
            if result:
                print(f"✅ Demo completed successfully!")
                print(f"   Output: {result['enhanced_path']}")
                print(f"   Comparison: {result['comparison_path']}")
            else:
                print("❌ Demo failed!")
                return 1
        else:
            print("❌ Demo mode requires single input file")
            return 1

        return 0

    # Full processing mode
    try:
        # Initialize processor
        print("🔧 Initializing processor...")
        processor = UniversalGANHTRProcessor(
            model_path=args.model_path,
            checkpoint_name=args.checkpoint_name,
            patch_size=tuple(args.patch_size),
            overlap=args.overlap,
            strategy=args.strategy,
            blend_method=args.blend_method,
            batch_size=args.batch_size
        )

        # Process input
        if args.input:
            print(f"🎯 Processing single image: {args.input}")
            result = processor.process_image(args.input, args.output)

            print(f"\n✅ Processing completed!")
            print(f"   Input: {result['input_path']}")
            print(f"   Output: {result['output_path']}")
            print(f"   Comparison: {result['comparison_path']}")
            print(f"   Patches: {result['n_patches']}")
            print(f"   PSNR: {result['reconstruction_metrics']['psnr']:.2f} dB")
            if result['reconstruction_metrics']['ssim'] is not None:
                print(f"   SSIM: {result['reconstruction_metrics']['ssim']:.4f}")

        else:
            print(f"📁 Processing directory: {args.input_dir}")
            results = processor.process_directory(args.input_dir, args.output_dir)

            print(f"\n✅ Batch processing completed!")
            print(f"   Total processed: {len(results)} images")

            if results:
                avg_psnr = sum(r['reconstruction_metrics']['psnr'] for r in results) / len(results)
                print(f"   Average PSNR: {avg_psnr:.2f} dB")

                ssim_values = [r['reconstruction_metrics']['ssim'] for r in results if r['reconstruction_metrics']['ssim'] is not None]
                if ssim_values:
                    avg_ssim = sum(ssim_values) / len(ssim_values)
                    print(f"   Average SSIM: {avg_ssim:.4f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    print(f"\n🎉 All operations completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())