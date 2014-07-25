# This sample configuration may be contrasted with "sample_config_full.py".
# Here, the vast majority of settings are not specified and are therefore set
# to the defaults.

config = {
    'sampler': {
      'subsample': 2, # downsampling 128 1ms to 64 2ms
      'resample_cache': 2,
      'time_dimension': 0,
      },
    'nets': [],
    }

ladder = [[0.1, 5,  2000,  5.],
          [0.3, 10, 2000,  2.],
          [0.5, 20, 2000,  2.],
          [0.7, 25, 4000,  1.0],
          [0.9, 30, 10000, 0.5],
          [1.0, 35, 40000, 0.5]]

for lam, maxit, num_iterations, target_angle in ladder:
  config['nets'].append({
      'num_iterations': num_iterations,
      'target_angle': target_angle,
      'max_angle': target_angle * 2,
      'inference_settings': {
        'lam': lam,
        'maxit': maxit
        }
      })
