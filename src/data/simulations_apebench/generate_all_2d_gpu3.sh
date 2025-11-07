python simulation.py --dimension 2 --pde ks --test_set --out_name ks_test --num_sims 50 --gpu_id 0 --low-res
python simulation.py --dimension 2 --pde gs_alpha --test_set --out_name gs_alpha_test --num_sims 30 --gpu_id 0 --low-res
python simulation.py --dimension 2 --pde gs_beta --test_set --out_name gs_beta_test --num_sims 30 --gpu_id 0 --low-res
python simulation.py --dimension 2 --pde gs_gamma --test_set --out_name gs_gamma_test --num_sims 30 --gpu_id 0 --low-res
python simulation.py --dimension 2 --pde gs_epsilon --test_set --out_name gs_epsilon_test --num_sims 30 --gpu_id 0 --low-res
python simulation.py --dimension 2 --pde decay_turb --test_set --out_name decay_turb_test --num_sims 50 --gpu_id 0 --low-res
python simulation.py --dimension 2 --pde kolm_flow --test_set --out_name kolm_flow_test --num_sims 50 --gpu_id 0 --low-res