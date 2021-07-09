from train import train, parse_args

def main(args):
    # Invoke subcommand.
    if args.command == "train":
        train(args, build_train_graph=build_train_graph)
    elif args.command == "compress":
        print(args)
        if not args.output_file:
            args.output_file = args.input_file + ".tfci"
        compress(args)
        # compress_est_ideal_rate(args)
    elif args.command == "decompress":
        print(args)
        if not args.output_file:
            args.output_file = args.input_file + ".png"
        decompress(args)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
