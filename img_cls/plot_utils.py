model_map = {'msdnet': 'MSDNet', 'dvit': 'DViT', 'l2wden': 'L2W-DEN', 'dynperc': 'Dyn-Perc', 'calib-msdnet/la-mie': 'Cal-MSDNet'}
color_map = {'msdnet': 'tab:blue', 'dvit': 'tab:orange', 'l2wden': 'tab:green', 'dynperc': 'tab:red', 'calib-msdnet/la-mie': 'tab:purple'}
risk_map = {'prediction-gt-gap': '$\mathcal{R}^G (\hat{y}) (0\!-\!1)$', 'confidence-brier': '$\mathcal{R}^G (\hat{p})$ (Brier)', 
            'prediction-consistency': '$\mathcal{R}^C (\hat{y}) (0\!-\!1)$', 'confidence-hellinger': '$\mathcal{R}^C(\hat{p})$ (Hellinger)', 
            'confidence-brier-top-pred': '$\mathcal{R}^C (\hat{p})$ (Brier)'}
ls_map = {"crc": "-", "ucb-wsr": "--"}