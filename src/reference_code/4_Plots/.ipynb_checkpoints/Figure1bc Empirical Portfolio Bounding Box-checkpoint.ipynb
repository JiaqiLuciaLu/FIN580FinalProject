{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.colors as mcolors\n",
    "import pandas as pd\n",
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.collections as mc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_out = \"../../Data/plot/Conditional_cutoffs\"\n",
    "if not os.path.exists(plot_out):\n",
    "    os.mkdir(plot_out)\n",
    "\n",
    "feats = ['LME', 'OP']\n",
    "filename = '_'.join(feats)\n",
    "feats_min = [np.array(pd.read_csv(\"../../Data/tree_portfolio_quantile/\"+filename+\"/level_all_\"+feat+\"_min.csv\",header=0)) for feat in feats]\n",
    "feats_max = [np.array(pd.read_csv(\"../../Data/tree_portfolio_quantile/\"+filename+\"/level_all_\"+feat+\"_max.csv\",header=0)) for feat in feats]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 432x432 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from matplotlib import colors as mcolors\n",
    "colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)\n",
    "\n",
    "lines = []\n",
    "c = []\n",
    "for i in range(16):\n",
    "    for j in range(16):\n",
    "        k = i * 16 + j\n",
    "        lines.append([(feats_min[0][0,k], feats_min[1][0,k]), (feats_min[0][0,k], feats_max[1][0,k])])\n",
    "        c.append(list(colors.items())[i][0])\n",
    "        lines.append([(feats_min[0][0,k], feats_min[1][0,k]), (feats_max[0][0,k], feats_min[1][0,k])])\n",
    "        c.append(list(colors.items())[i][0])\n",
    "        lines.append([(feats_max[0][0,k], feats_min[1][0,k]), (feats_max[0][0,k], feats_max[1][0,k])])\n",
    "        c.append(list(colors.items())[i][0])\n",
    "        lines.append([(feats_min[0][0,k], feats_max[1][0,k]), (feats_max[0][0,k], feats_max[1][0,k])])\n",
    "        c.append(list(colors.items())[i][0])\n",
    "        \n",
    "lc = mc.LineCollection(lines, color = c, linewidths=1, linestyle='dashed')\n",
    "fig, ax = plt.subplots(figsize=(6, 6))\n",
    "ax.add_collection(lc)\n",
    "ax.autoscale()\n",
    "plt.xlabel('LME')\n",
    "plt.ylabel('OP')\n",
    "\n",
    "plt.savefig(os.path.join(plot_out, filename + '_tree.png'), dpi=300)\n",
    "plt.clf()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 432x432 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)\n",
    "\n",
    "lines = []\n",
    "c = []\n",
    "\n",
    "i = 1\n",
    "lines.append([(0.25, 0), (0.25, 1)])\n",
    "c.append(list(colors.items())[i][0])\n",
    "lines.append([(0.5, 0), (0.5, 1)])\n",
    "c.append(list(colors.items())[i][0])\n",
    "lines.append([(0.75, 0), (0.75, 1)])\n",
    "c.append(list(colors.items())[i][0])\n",
    "\n",
    "i = 2\n",
    "lines.append([(0, 0.25), (1, 0.25)])\n",
    "c.append(list(colors.items())[i][0])\n",
    "lines.append([(0, 0.5), (1, 0.5)])\n",
    "c.append(list(colors.items())[i][0])\n",
    "lines.append([(0, 0.75), (1, 0.75)])\n",
    "c.append(list(colors.items())[i][0])\n",
    "              \n",
    "lc = mc.LineCollection(lines, color = c, linewidths=1, linestyle='dashed')\n",
    "fig, ax = plt.subplots(figsize=(6, 6))\n",
    "ax.add_collection(lc)\n",
    "ax.autoscale()\n",
    "plt.xlabel('LME')\n",
    "plt.ylabel('OP')\n",
    "\n",
    "plt.savefig(os.path.join(plot_out, filename + '_ds.png'), dpi=300)\n",
    "plt.clf()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
