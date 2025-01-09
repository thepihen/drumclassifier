from classifier import DrumClassifier
import argparse

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, required=True, help='Input audio file path')
    parser.add_argument('-w','--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('-c','--cfg', type=str, default='cfg.yaml', help="Model cfg path (default='cfg.yaml')")
    args = parser.parse_args()
    clsf = DrumClassifier(weights_path=args.weights, cfg_path=args.cfg)
    print(f"Your sample contains: {clsf.classify_from_path(args.input, output_text_label=True)}")
    