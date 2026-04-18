{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### This part depends on yearly data chunk file generated from all 10 characteristics"
   ]
  },
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
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "plot_out = \"../../Data/plot/EmpiricalDist/Char10/\"\n",
    "if not os.path.exists(plot_out):\n",
    "    os.mkdir(plot_out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "feats_list = ['LME','BEME','r12_2','OP','Investment', 'ST_Rev', 'LT_Rev','AC','IdioVol','LTurnover']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1963\n",
      "1964\n",
      "1965\n",
      "1966\n",
      "1967\n",
      "1968\n",
      "1969\n",
      "1970\n",
      "1971\n",
      "1972\n",
      "1973\n",
      "1974\n",
      "1975\n",
      "1976\n",
      "1977\n",
      "1978\n",
      "1979\n",
      "1980\n",
      "1981\n",
      "1982\n",
      "1983\n",
      "1984\n",
      "1985\n",
      "1986\n",
      "1987\n",
      "1988\n",
      "1989\n",
      "1990\n",
      "1991\n",
      "1992\n",
      "1993\n",
      "1994\n",
      "1995\n",
      "1996\n",
      "1997\n",
      "1998\n",
      "1999\n",
      "2000\n",
      "2001\n",
      "2002\n",
      "2003\n",
      "2004\n",
      "2005\n",
      "2006\n",
      "2007\n",
      "2008\n",
      "2009\n",
      "2010\n",
      "2011\n",
      "2012\n",
      "2013\n",
      "2014\n",
      "2015\n",
      "2016\n"
     ]
    }
   ],
   "source": [
    "grid = 50\n",
    "fine_grid = 10\n",
    "\n",
    "x = np.linspace(0, 1, grid * fine_grid)\n",
    "y = np.linspace(0, 1, grid * fine_grid)\n",
    "counts = np.zeros((45, grid*fine_grid, grid*fine_grid))\n",
    "\n",
    "for year in range(1963, 2017):\n",
    "    print(year)\n",
    "    data = pd.read_csv(\"../../Data/data_chunk_files_quantile/Char10/y\" + str(year) + \".csv\",header=0)\n",
    "    \n",
    "    data_cols = np.zeros((data.shape[0], 10))\n",
    "    \n",
    "    for i in range(10):\n",
    "        data_cols[:, i] = np.floor(np.array(data[feats_list[i]]) * grid)\n",
    "        data_cols[:, i][data_cols[:, i] == grid] = (grid - 1)\n",
    "        \n",
    "    data_cols = data_cols.astype(int)\n",
    "    \n",
    "    for i in range(data_cols.shape[0]):\n",
    "        l = 0\n",
    "        for j in range(9):\n",
    "            for k in range(j + 1, 10):\n",
    "                counts[l, (fine_grid*data_cols[i, j]):(fine_grid*data_cols[i, j]+fine_grid), (fine_grid*data_cols[i, k]):(fine_grid*data_cols[i, k]+fine_grid)] += 1\n",
    "                l += 1\n",
    "        \n",
    "                    \n",
    "x, y = np.meshgrid(x, y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/jasonzy/anaconda3/lib/python3.6/site-packages/matplotlib/pyplot.py:528: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).\n",
      "  max_open_warning, RuntimeWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b789128>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b90ada0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef797d4860>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b8b4ef0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a4b9c50>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bafff98>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc15898>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a593b70>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b8680f0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a38fb70>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b869f28>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bacc4a8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a727278>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7baaadd8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a690278>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b7ff390>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bacc898>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a458198>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc1c390>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a690940>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a35e780>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a6fa358>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bacccf8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a4ca4a8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b842400>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bb1cba8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef82677198>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7968a2b0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b8a4668>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a718c50>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef797aeda0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7d056f98>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7968add8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bbed940>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a653d30>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a57d6a0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b65bf60>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a1dec88>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7966e198>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef797c11d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a1bca58>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b83f128>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bb16128>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bd67e10>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a4a9a90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from matplotlib import ticker, cm\n",
    "\n",
    "def fmt(x, pos):\n",
    "    a, b = '{:.2e}'.format(x).split('e')\n",
    "    b = int(b)\n",
    "    return r'${} \\times 10^{{{}}}$'.format(a, b)\n",
    "\n",
    "l = 0\n",
    "total = np.sum(counts[0,:,:])\n",
    "counts = counts/total * 100\n",
    "cbar_min = -4.5\n",
    "cbar_max = -2.5\n",
    "counts[counts < (10 ** cbar_min)] = 10 ** cbar_min\n",
    "counts[counts > (10 ** cbar_max)] = 10 ** cbar_max\n",
    "\n",
    "for j in range(9):\n",
    "    for k in range(j + 1, 10):\n",
    "        plt.figure(figsize=(8,5))\n",
    "        im = plt.contourf(x, y, counts[l,:,:], locator=ticker.LogLocator(), levels = np.logspace(cbar_min, cbar_max, 50))\n",
    "        plt.xlabel(feats_list[j])\n",
    "        plt.ylabel(feats_list[k])\n",
    "        cbar = plt.colorbar(im, format=ticker.FuncFormatter(fmt))\n",
    "        cbar.ax.set_ylabel('Dist (%)')\n",
    "        plt.savefig(os.path.join(plot_out + 'Counts/', feats_list[j] + '_' + feats_list[k] + '.png'))\n",
    "        plt.clf()\n",
    "        l += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1963\n",
      "1964\n",
      "1965\n",
      "1966\n",
      "1967\n",
      "1968\n",
      "1969\n",
      "1970\n",
      "1971\n",
      "1972\n",
      "1973\n",
      "1974\n",
      "1975\n",
      "1976\n",
      "1977\n",
      "1978\n",
      "1979\n",
      "1980\n",
      "1981\n",
      "1982\n",
      "1983\n",
      "1984\n",
      "1985\n",
      "1986\n",
      "1987\n",
      "1988\n",
      "1989\n",
      "1990\n",
      "1991\n",
      "1992\n",
      "1993\n",
      "1994\n",
      "1995\n",
      "1996\n",
      "1997\n",
      "1998\n",
      "1999\n",
      "2000\n",
      "2001\n",
      "2002\n",
      "2003\n",
      "2004\n",
      "2005\n",
      "2006\n",
      "2007\n",
      "2008\n",
      "2009\n",
      "2010\n",
      "2011\n",
      "2012\n",
      "2013\n",
      "2014\n",
      "2015\n",
      "2016\n"
     ]
    }
   ],
   "source": [
    "grid = 10\n",
    "fine_grid = 10\n",
    "\n",
    "x = np.linspace(0, 1, grid * fine_grid)\n",
    "y = np.linspace(0, 1, grid * fine_grid)\n",
    "counts_ret = np.zeros((45, grid*fine_grid, grid*fine_grid))\n",
    "avg_ret = np.zeros((45, grid*fine_grid, grid*fine_grid))\n",
    "\n",
    "for year in range(1963, 2017):\n",
    "    print(year)\n",
    "    data = pd.read_csv(\"../../Data/data_chunk_files_quantile/Char10/y\" + str(year) + \".csv\",header=0)\n",
    "    \n",
    "    data_cols = np.zeros((data.shape[0], 10))\n",
    "    \n",
    "    for i in range(10):\n",
    "        data_cols[:, i] = np.floor(np.array(data[feats_list[i]]) * grid)\n",
    "        data_cols[:, i][data_cols[:, i] == grid] = (grid - 1)\n",
    "    \n",
    "    ret = np.log(np.array(data['ret']) + 1)\n",
    "    \n",
    "    data_cols = data_cols.astype(int)\n",
    "    \n",
    "    for i in range(data_cols.shape[0]):\n",
    "        l = 0\n",
    "        for j in range(9):\n",
    "            for k in range(j + 1, 10):\n",
    "                counts_ret[l, (fine_grid*data_cols[i, j]):(fine_grid*data_cols[i, j]+fine_grid), (fine_grid*data_cols[i, k]):(fine_grid*data_cols[i, k]+fine_grid)] += 1\n",
    "                avg_ret[l, (fine_grid*data_cols[i, j]):(fine_grid*data_cols[i, j]+fine_grid), (fine_grid*data_cols[i, k]):(fine_grid*data_cols[i, k]+fine_grid)] += ret[i]\n",
    "                l += 1\n",
    "        \n",
    "                    \n",
    "x, y = np.meshgrid(x, y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/jasonzy/anaconda3/lib/python3.6/site-packages/matplotlib/pyplot.py:528: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).\n",
      "  max_open_warning, RuntimeWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a4403c8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b6f8978>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef826d8630>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a997d30>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a2c35c0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a3fc9e8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a66f6a0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b681518>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a740080>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b86bb38>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc74a90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a4ce668>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef82677e48>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7baccc50>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a3fddd8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b83a8d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a410588>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a407668>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b78ccf8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7d048d30>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc85470>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b6fd668>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bd7f198>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b838cc0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a7494a8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b6e6710>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a2c22b0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b9dd710>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc1c7b8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc9a550>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a690b70>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a4e2278>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7babb2e8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b858f98>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b639128>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a4bcb70>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef79647550>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a64ce10>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a737da0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a271160>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a4cea90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a5757f0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7baa9438>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bab5a90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bbcceb8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "l = 0\n",
    "ret = avg_ret/ counts_ret\n",
    "for j in range(9):\n",
    "    for k in range(j + 1, 10):\n",
    "        plt.figure(figsize=(8,5))\n",
    "        im = plt.contourf(x, y, ret[l,:,:], levels = np.linspace(np.min(ret), np.max(ret), 50))\n",
    "        plt.xlabel(feats_list[j])\n",
    "        plt.ylabel(feats_list[k])\n",
    "        cbar = plt.colorbar(im)\n",
    "        cbar.ax.set_ylabel('Avg Log Ret')\n",
    "        plt.savefig(os.path.join(plot_out + 'Avg Log Ret/', feats_list[j] + '_' + feats_list[k] + '.png'))\n",
    "        plt.clf()\n",
    "        l += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1963\n",
      "1964\n",
      "1965\n",
      "1966\n",
      "1967\n",
      "1968\n",
      "1969\n",
      "1970\n",
      "1971\n",
      "1972\n",
      "1973\n",
      "1974\n",
      "1975\n",
      "1976\n",
      "1977\n",
      "1978\n",
      "1979\n",
      "1980\n",
      "1981\n",
      "1982\n",
      "1983\n",
      "1984\n",
      "1985\n",
      "1986\n",
      "1987\n",
      "1988\n",
      "1989\n",
      "1990\n",
      "1991\n",
      "1992\n",
      "1993\n",
      "1994\n",
      "1995\n",
      "1996\n",
      "1997\n",
      "1998\n",
      "1999\n",
      "2000\n",
      "2001\n",
      "2002\n",
      "2003\n",
      "2004\n",
      "2005\n",
      "2006\n",
      "2007\n",
      "2008\n",
      "2009\n",
      "2010\n",
      "2011\n",
      "2012\n",
      "2013\n",
      "2014\n",
      "2015\n",
      "2016\n"
     ]
    }
   ],
   "source": [
    "grid = 5\n",
    "fine_grid = 1\n",
    "\n",
    "x = np.linspace(0, 1, grid * fine_grid)\n",
    "y = np.linspace(0, 1, grid * fine_grid)\n",
    "counts_ret = np.zeros((45, grid*fine_grid, grid*fine_grid))\n",
    "avg_ret = np.zeros((45, grid*fine_grid, grid*fine_grid))\n",
    "\n",
    "for year in range(1963, 2017):\n",
    "    print(year)\n",
    "    data = pd.read_csv(\"../../Data/data_chunk_files_quantile/Char10/y\" + str(year) + \".csv\",header=0)\n",
    "    \n",
    "    data_cols = np.zeros((data.shape[0], 10))\n",
    "    \n",
    "    for i in range(10):\n",
    "        data_cols[:, i] = np.floor(np.array(data[feats_list[i]]) * grid)\n",
    "        data_cols[:, i][data_cols[:, i] == grid] = (grid - 1)\n",
    "    \n",
    "    ret = np.log(np.array(data['ret']) + 1)\n",
    "    \n",
    "    data_cols = data_cols.astype(int)\n",
    "    \n",
    "    for i in range(data_cols.shape[0]):\n",
    "        l = 0\n",
    "        for j in range(9):\n",
    "            for k in range(j + 1, 10):\n",
    "                counts_ret[l, (fine_grid*data_cols[i, j]):(fine_grid*data_cols[i, j]+fine_grid), (fine_grid*data_cols[i, k]):(fine_grid*data_cols[i, k]+fine_grid)] += 1\n",
    "                avg_ret[l, (fine_grid*data_cols[i, j]):(fine_grid*data_cols[i, j]+fine_grid), (fine_grid*data_cols[i, k]):(fine_grid*data_cols[i, k]+fine_grid)] += ret[i]\n",
    "                l += 1\n",
    "        \n",
    "                    \n",
    "x, y = np.meshgrid(x, y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "l = 0\n",
    "ret = avg_ret/ counts_ret\n",
    "for j in range(9):\n",
    "    for k in range(j + 1, 10):\n",
    "        cond_ret = np.zeros([6, 5])\n",
    "        \n",
    "        for i in range(5):\n",
    "            cond_ret[i,:] = ret[l, i,:]\n",
    "        \n",
    "        cond_ret[5, :] = np.sum(avg_ret[l,:,:], axis = 0)/np.sum(counts_ret[l,:,:], axis = 0)\n",
    "        \n",
    "        my_df = pd.DataFrame(cond_ret)\n",
    "        my_df.to_csv(\"../../Data/ds_conditional_ret/\"+ feats_list[j] + '_' + feats_list[k] +\".csv\", index=False, header=False)\n",
    "\n",
    "        l += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/jasonzy/anaconda3/lib/python3.6/site-packages/matplotlib/pyplot.py:528: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).\n",
      "  max_open_warning, RuntimeWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7f9e1da0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a374f60>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a13fb00>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b869080>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7967eeb8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b8ffba8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc74240>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a38f278>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a2ae9b0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bba4438>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b6e45f8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a340d68>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a3ecbe0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc85470>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a2ae5f8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a271160>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a4ceeb8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b8d4c18>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a2bc748>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bbe45c0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc856d8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b8b8b38>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef797c1320>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b8b4518>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc850f0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a37e5c0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7ef6e518>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7ee3c9b0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a3f1358>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bd81898>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b4f8cf8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a722780>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bc1c8d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a3b9668>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7bb9b908>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a374ac8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a42ff60>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a14d5c0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b8d4be0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b8c7080>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7a722eb8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b68c3c8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7b4f8ac8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7967ea90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fef7966eb38>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "l = 0\n",
    "ret = avg_ret/ counts_ret\n",
    "for j in range(9):\n",
    "    for k in range(j + 1, 10):\n",
    "        plt.figure(figsize=(8,5))\n",
    "        ax = plt.axes()\n",
    "        x = np.linspace(1, 5, 5)\n",
    "        for i in range(5):\n",
    "            ax.plot(x, ret[l, i,:], label= feats_list[j] + '_' + str(i+1))\n",
    "        \n",
    "        ax.plot(x, np.sum(avg_ret[l,:,:], axis = 0)/np.sum(counts_ret[l,:,:], axis = 0), label= 'Unconditional')\n",
    "        \n",
    "        plt.xlabel(feats_list[k])\n",
    "        plt.xticks(x)\n",
    "        plt.ylabel('Avg Log Return')\n",
    "        plt.legend(loc = 1)\n",
    "        \n",
    "        plt.savefig(os.path.join(plot_out + 'Cond_Avg_Ret/', feats_list[j] + '_' + feats_list[k] + '.png'))\n",
    "        plt.clf()\n",
    "        l += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.00676337,  0.00072855,  0.00368344,  0.00434408,  0.00411891])"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.sum(avg_ret[l,:,:], axis = 0)/np.sum(counts_ret[l,:,:], axis = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "l=0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fe223af7240>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "l = 0\n",
    "for j in range(9):\n",
    "    for k in range(j + 1, 10):\n",
    "        im = plt.contourf(x, y, np.log(counts[l,:,:]))\n",
    "        plt.xlabel(feats_list[k])\n",
    "        plt.ylabel('Avg Log Return')\n",
    "        cbar = plt.colorbar(im)\n",
    "        cbar.ax.set_ylabel('Log Counts')\n",
    "        plt.savefig(os.path.join(plot_out + 'Log Counts/', feats_list[j] + '_' + feats_list[k] + '_log.png'))\n",
    "        plt.clf()\n",
    "        l += 1"
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
