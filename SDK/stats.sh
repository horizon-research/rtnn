dumpstats() {
  cat $1 | awk '{if ($2=="upload") t+=$6; if ($2=="total") t+=$5}END{print t}'
  #cat $1 | awk '{if ($2=="total") t+=$5}END{print t}'
}

### R/K N-BODY ###
rk_nbody() {
  for i in 1
  do
    ab=1
  
    for m in radius knn
    do
      for r in 1 2 8 10 32 128
      do
        echo -n $m" r "$r" "
        dumpstats "nbody-9m-p1-r$r-k50-$m-ab${ab}-run${i}.out"
      done
  
      for k in 1 4 16 32 50 64 128
      do
        echo -n $m" k "$k" "
        dumpstats "nbody-9m-p1-r32-k$k-$m-ab${ab}-run${i}.out"
      done
    done
  done
}

### R/K GRAPHICS ###
rk_graphics() {
  for i in 1
  do
    ab=1
  
    for m in radius knn
    do
      for k in 1 4 16 32 50 64 128
      do
        echo -n $m" k "$k" "
        dumpstats "buddha-p1-r0.05-k$k-$m-ab1-run${i}.out"
      done
  
      for r in 0.001 0.005 0.01 0.05 0.1 0.2
      do
        echo -n $m" r "$r" "
        dumpstats "buddha-p1-r$r-k50-$m-ab1-run${i}.out"
      done
    done
  done
}

### R/K KITTI ###
rk_kitti() {
  for i in 1
  do
    ab=1
  
    for m in radius knn
    do
      for r in 0.01 0.1 2 16 50
      do
        echo -n $m" r "$r" "
        dumpstats "kitti-6m-p1-r${r}-k50-$m-ab1-run${i}.out"
      done
  
      for k in 1 4 16 32 50 64 100 128 200
      do
        echo -n $m" k "$k" "
        dumpstats "kitti-6m-p1-r2-k${k}-$m-ab${ab}-run${i}.out"
      done
    done
  done
}

### ALL EVAL ###
all_eval() {
  for i in 1
  do
    ab=1
  
    echo -n "knn kitti-120k "
    dumpstats "kitti-120k-p1-r2-k50-knn-ab${ab}-run${i}.out"
    echo -n "knn kitti-1m "
    dumpstats "kitti-1m-p1-r2-k50-knn-ab${ab}-run${i}.out"
    echo -n "knn kitti-6m "
    dumpstats "kitti-6m-p1-r2-k50-knn-ab${ab}-run${i}.out"
    echo -n "knn kitti-12m "
    dumpstats "kitti-12m-p1-r2-k50-knn-ab${ab}-run${i}.out"
    echo -n "knn nbody-9m "
    dumpstats "nbody-9m-p1-r32-k50-knn-ab${ab}-run${i}.out"
    echo -n "knn nbody-10m "
    dumpstats "nbody-10m-p1-r32-k50-knn-ab${ab}-run${i}.out"
    echo -n "knn bunny "
    dumpstats "bunny-p1-r0.05-k50-knn-ab${ab}-run${i}.out"
    echo -n "knn dragon "
    dumpstats "dragon-p1-r2-k50-knn-ab${ab}-run${i}.out"
    echo -n "knn buddha "
    dumpstats "buddha-p1-r0.05-k50-knn-ab${ab}-run${i}.out"
  
    echo -n "radius kitti-120k "
    dumpstats "kitti-120k-p1-r2-k50-radius-ab${ab}-run${i}.out"
    echo -n "radius kitti-1m "
    dumpstats "kitti-1m-p1-r2-k50-radius-ab${ab}-run${i}.out"
    echo -n "radius kitti-6m "
    dumpstats "kitti-6m-p1-r2-k50-radius-ab${ab}-run${i}.out"
    echo -n "radius kitti-12m "
    dumpstats "kitti-12m-p1-r2-k50-radius-ab${ab}-run${i}.out"
    echo -n "radius kitti-25m "
    dumpstats "kitti-25m-p1-r2-k50-radius-ab${ab}-run${i}.out"
    echo -n "radius nbody-9m "
    dumpstats "nbody-9m-p1-r32-k50-radius-ab${ab}-run${i}.out"
    echo -n "radius nbody-10m "
    dumpstats "nbody-10m-p1-r32-k50-radius-ab${ab}-run${i}.out"
    echo -n "radius bunny "
    dumpstats "bunny-p1-r0.05-k50-radius-ab${ab}-run${i}.out"
    echo -n "radius dragon "
    dumpstats "dragon-p1-r2-k50-radius-ab${ab}-run${i}.out"
    echo -n "radius buddha "
    dumpstats "buddha-p1-r0.05-k50-radius-ab${ab}-run${i}.out"
  
  done
}

### BATCHING ###
batch_eval() {
  for m in knn radius
  do
    for i in $(seq 1 1 23)
    do
      echo -n "bunny "$m" "$i" "
      dumpstats "bunny-p1-r0.05-k50-$m-ab0-nb$i-run1.out"
    done
  
    for i in $(seq 1 1 10)
    do
      echo -n "dragon "$m" "$i" "
      dumpstats "dragon-p1-r2-k50-$m-ab0-nb$i-run1.out"
    done
  
    for i in $(seq 1 1 23)
    do
      echo -n "buddha "$m" "$i" "
      dumpstats "buddha-p1-r0.05-k50-$m-ab0-nb$i-run1.out"
    done
    
    for i in $(seq 1 1 46)
    do
      echo -n "nbody "$m" "$i" "
      dumpstats "nbody-9m-p1-r32-k50-$m-ab0-nb$i-run1.out"
    done
  
    for i in $(seq 1 1 26)
    do
      echo -n "120k "$m" "$i" "
      dumpstats "kitti-120k-p1-r2-k50-$m-ab0-nb$i-run1.out"
    done
  
    for i in $(seq 1 1 23)
    do
      echo -n "1m "$m" "$i" "
      dumpstats "kitti-1m-p1-r2-k50-$m-ab0-nb$i-run1.out"
    done
  
    for i in $(seq 1 1 18)
    do
      echo -n "6m "$m" "$i" "
      dumpstats "kitti-6m-p1-r2-k50-$m-ab0-nb$i-run1.out"
    done
  
    for i in $(seq 1 1 16)
    do
      echo -n "12m "$m" "$i" "
      dumpstats "kitti-12m-p1-r2-k50-$m-ab0-nb$i-run1.out"
    done
  done
}

### OPT SENSITIVITY ###
  opt_sen() {
  run() {
    echo "s "$1 "ps "$2 "p "$3 "ab "$4
    dumpstats "kitti-12m-s$1-ps$2-p$3-ab$4-r2-k50-$5-run$6.out"
    dumpstats "nbody-9m-s$1-ps$2-p$3-ab$4-r32-k50-$5-run$6.out"
    dumpstats "buddha-s$1-ps$2-p$3-ab$4-r0.05-k50-$5-run$6.out"
  }
  
  for i in 1
  do
    run 0 0 0 0 "radius" $i
    run 2 1 0 0 "radius" $i
    run 2 1 1 0 "radius" $i
    run 2 1 1 1 "radius" $i
    run 0 0 0 0 "knn" $i
    run 2 1 0 0 "knn" $i
    run 2 1 1 0 "knn" $i
    run 2 1 1 1 "knn" $i
  done
}

rk_kitti
rk_nbody
rk_graphics
exit
all_eval
batch_eval
opt_sen

