def main(args):
    logger = get_logger()
    # Ensure directories are created
    os.makedirs(args.store_inv_hessian_dir, exist_ok=True)

    percdamp = 0.01  # Damping factor
    hessian_files = [f for f in os.listdir(args.load_hessian_dir) if f.endswith('.pt')]
    logger.info(f"Found Hessian files: {hessian_files}")

    for hessian_file in (pbar := tqdm.tqdm(hessian_files, desc="Inverting Hessian")):
        hessian_path = os.path.join(args.load_hessian_dir, hessian_file)
        try:
            hessian, mu = load_hessian(hessian_path, pbar=pbar, logger=logger)
            logger.info(f"Loaded Hessian from {hessian_path}")
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            hessian = hessian.to(dev)

            zero_idx = torch.diag(hessian) == 0
            hessian[zero_idx, zero_idx] = 1

            # Permutation logic
            perm = torch.argsort(torch.diag(hessian), descending=True).to(dev)
            if args.enable_perm:
                hessian = hessian[perm][:, perm]

            # Add damping
            damp = percdamp * torch.mean(torch.diag(hessian))
            diag = torch.arange(hessian.shape[0], device=dev)
            hessian[diag, diag] += damp

            # Inverse Hessian computation
            hessian = torch.linalg.cholesky(hessian)
            hessian = torch.cholesky_inverse(hessian)
            hessian = torch.linalg.cholesky(hessian, upper=True)
            inv_hessian = hessian

            # Save inverted Hessian
            save_path = os.path.join(args.store_inv_hessian_dir, hessian_file)
            if not args.enable_perm:
                perm = torch.arange(inv_hessian.shape[0])

            torch.save({'invH': inv_hessian.to('cpu'),
                        'perm': perm.to('cpu'),
                        'zero_idx': zero_idx.to('cpu')}, save_path)
            logger.info(f"Saved inverted Hessian to {save_path}")
            pbar.set_postfix_str(f"Saved inverted Hessian to {save_path}")

        except Exception as e:
            logger.error(f"Error processing {hessian_file}: {e}", exc_info=True)
