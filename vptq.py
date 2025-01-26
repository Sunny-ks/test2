To quantize your fine-tuned model to **2 bits**, **4 bits**, and **8 bits** using the VPTQ algorithm, you need to adjust the parameters `--num_centroids` and `--num_res_centroids` in the `run_vptq.py` command. These parameters determine the quantization levels for main and residual centroids.

Here are the commands for each bitwidth:

---

### **1. Quantize to 2 bits**
For 2-bit quantization:
- **Main centroids:** \(2^2 = 4\)
- **Residual centroids:** \(2^0 = 1\) (no residual quantization)

Command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_vptq.py \
    --model_name <your-finetuned-model-name> \
    --output_dir outputs/<your-model-name>-2bits/ \
    --vector_lens -1 8 \
    --group_num 1 \
    --num_centroids -1 4 \
    --num_res_centroids -1 1 \
    --npercent 0 \
    --blocksize 128 \
    --new_eval \
    --seq_len 8192 \
    --kmeans_mode hessian \
    --num_gpus 8 \
    --enable_perm \
    --enable_norm \
    --save_model \
    --save_packed_model \
    --hessian_path <hessian-file-path> \
    --inv_hessian_path <inv-hessian-file-path> \
    --ktol 1e-5 --kiter 100
```

---

### **2. Quantize to 4 bits**
For 4-bit quantization:
- **Main centroids:** \(2^4 = 16\)
- **Residual centroids:** \(2^4 = 16\)

Command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_vptq.py \
    --model_name <your-finetuned-model-name> \
    --output_dir outputs/<your-model-name>-4bits/ \
    --vector_lens -1 8 \
    --group_num 1 \
    --num_centroids -1 16 \
    --num_res_centroids -1 16 \
    --npercent 0 \
    --blocksize 128 \
    --new_eval \
    --seq_len 8192 \
    --kmeans_mode hessian \
    --num_gpus 8 \
    --enable_perm \
    --enable_norm \
    --save_model \
    --save_packed_model \
    --hessian_path <hessian-file-path> \
    --inv_hessian_path <inv-hessian-file-path> \
    --ktol 1e-5 --kiter 100
```

---

### **3. Quantize to 8 bits**
For 8-bit quantization:
- **Main centroids:** \(2^8 = 256\)
- **Residual centroids:** \(2^8 = 256\)

Command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_vptq.py \
    --model_name <your-finetuned-model-name> \
    --output_dir outputs/<your-model-name>-8bits/ \
    --vector_lens -1 8 \
    --group_num 1 \
    --num_centroids -1 256 \
    --num_res_centroids -1 256 \
    --npercent 0 \
    --blocksize 128 \
    --new_eval \
    --seq_len 8192 \
    --kmeans_mode hessian \
    --num_gpus 8 \
    --enable_perm \
    --enable_norm \
    --save_model \
    --save_packed_model \
    --hessian_path <hessian-file-path> \
    --inv_hessian_path <inv-hessian-file-path> \
    --ktol 1e-5 --kiter 100
```

---

### **General Notes**
1. Replace placeholders like `<your-finetuned-model-name>` and `<hessian-file-path>` with actual values:
   - `--model_name`: Path to your fine-tuned model.
   - `--hessian_path` and `--inv_hessian_path`: Path to precomputed Hessian and inverse Hessian files. You can skip these if you don't have them, but accuracy might be affected.

2. Adjust the number of GPUs (`--num_gpus`) or GPU IDs in `CUDA_VISIBLE_DEVICES` based on your hardware availability.

3. Logs can be checked for quantization accuracy:
   ```bash
   cat outputs/<your-model-name>-Xbits/{your_path}/log/0.log
   ```

Let me know if you want me to explain or customize any part of these commands!
