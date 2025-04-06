for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    sbatch /home/myang4/TinyZero/examples/data_preprocess/math_difficulty.sh $((28672 + i * 64)) $((28672 + (i + 1) * 64))
done

