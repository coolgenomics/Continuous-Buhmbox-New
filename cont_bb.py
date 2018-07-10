import os.path
import numpy as np
import time
from scipy.stats import rankdata
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt


def generate_pss_model_generalized(num_phenos=2, eff_size=0.1, eff_afreq=0.5,num_snps=np.array([[0, 50], [50, 0]])):
    all_ps = []
    all_betas = []

    p_low=eff_afreq
    p_high=eff_afreq
    b_loc = eff_size
    b_scale = 0.0 # no distribution: just take the mean every time  
    
    if num_snps.shape != tuple([2] * num_phenos):
        raise ValueError("Invalid num_snps passed to generate_pss_model_generalized")
    
    for index, val in np.ndenumerate(num_snps):
        name_counter = 0
        n = int(val)
        p = np.random.uniform(low=p_low,high=p_high, size=n)
        beta = np.random.normal(loc=b_loc,scale=b_scale, size=(n, num_phenos)) * np.array(index).reshape(1, num_phenos)
        all_ps.append(p)
        all_betas.append(beta)
    return np.hstack(all_ps), np.vstack(all_betas)

def calc_var_from_geno(snp_props):
    snp_ps, snp_betas = snp_props
    num_snps, num_phenos = snp_betas.shape
    snp_ps_all = np.repeat(snp_ps.reshape(num_snps, 1), num_phenos, axis=1)
    return np.sum(snp_betas**2 * 2 * snp_ps_all * (1 - snp_ps_all), axis=0)

def calc_pop_var(pop):
    _, phenos, _ = pop
    return np.var(phenos, axis=0)

def calc_pop_gen_var(pop, snp_props):
    snp_ps, snp_betas = snp_props
    genos, phenos, _ = pop
    return np.var(np.dot(genos, snp_betas), axis=0)

def generate_hetero_population(snps, num_inds=10000, h=1):
    """
    snps=[snp_ps, snp_betas]
    snp_ps: numpy length num_snps array with rafs
    snp_betas: numpy (num_snps, num phenos) matrix with betas
    """
    geno, pheno, _ = generate_population(snps, num_inds=num_inds, h=h)
    
    alts = pheno[:, -1] > pheno[:, -2]
    alts_float = alts.astype(float)
    new_pheno = (pheno[:, -1] * alts_float) + (pheno[:, -2] * (1 - alts_float))
    real_phenos = np.hstack((pheno[:, :-2], new_pheno.reshape(num_inds, 1)))
    return geno, real_phenos, alts

def generate_population(snps, num_inds=10000, h=1):
    """
    snps=[snp_ps, snp_betas]
    snp_ps: numpy length num_snps array with rafs
    snp_betas: numpy (num_snps, num phenos) matrix with betas
    """
    snp_ps, snp_betas = snps
    assert len(snp_ps) == len(snp_betas)
    num_snps = len(snp_ps)
    assert num_snps > 0
    num_phenos = len(snp_betas[0])

    # sample SNPs according to SNP props
    randoms = np.random.rand(num_inds, num_snps, 1)
    snp_ps_all = np.repeat(snp_ps.reshape(1, num_snps, 1), num_inds, axis=0)
    geno = (randoms < snp_ps_all**2.0).astype(float) + (randoms < snp_ps_all**2.0 + 2.0*snp_ps_all*(1.0-snp_ps_all)).astype(float)
    assert geno.shape == (num_inds, num_snps, 1)
    betas_all = np.repeat(snp_betas.reshape(1, num_snps, num_phenos), num_inds, axis=0)
    pheno = np.sum(np.repeat(geno, num_phenos, axis=2) * betas_all, axis=1)
    assert pheno.shape == (num_inds, num_phenos)
    geno = geno.reshape(num_inds, num_snps)
    genetic_var = calc_var_from_geno(snps)
    sigma = np.sqrt((1-h)/h * genetic_var)
    pheno = pheno + np.random.normal(loc=np.zeros(num_phenos), scale=sigma, size=(num_inds, num_phenos))
    return geno, pheno, np.zeros(num_inds)

def buhmbox(cases,controls,clist,snp_props):
    """
    cases, controls: Numpy array where each row is an indiv and each col is a snp
    clist: tuple of indices that are the snps for DB
    snp_props: 
    """
    num_snps = len(clist)
    snp_cases_all = cases[:,clist]
    snp_controls_all = controls[:,clist]
    num_cases = snp_cases_all.shape[0]
    num_controls = snp_controls_all.shape[0]
    percent_holdout = 0.05
    num_holdout_cases = int(0.05 * num_cases)
    num_holdout_controls = int(0.05 * num_controls)
    num_used_cases = num_cases - num_holdout_cases
    num_used_controls = num_controls - num_holdout_cases

    snp_cases_hold = snp_cases_all[:num_holdout_cases]
    snp_cases = snp_cases_all[num_holdout_cases:]
    snp_controls_hold = snp_controls_all[:num_holdout_controls]
    snp_controls = snp_controls_all[num_holdout_controls:]

    N = float(len(snp_cases))
    Np = float(len(snp_controls))
    R = np.corrcoef(snp_cases.T)
    Rp = np.corrcoef(snp_controls.T)
    Y = np.sqrt(N*Np/(Np-N)) * (R-Rp)
    
    pi_cases = np.sum(snp_cases_hold, axis=0) / (2*snp_cases_hold.shape[0])
    pi_controls = np.sum(snp_controls_hold, axis=0) / (2*snp_controls_hold.shape[0])
    gamma = pi_cases/(1-pi_cases) / (pi_controls/(1-pi_controls))
    
    # calculate SBB
    elem1 = np.sqrt(pi_controls*(1-pi_controls))
    elem2 = gamma-1
    elem3 = elem2 * pi_controls + 1
    mat1 = np.sqrt(np.dot(elem1.reshape((num_snps, 1)), elem1.reshape((1, num_snps))))
    mat2 = np.dot(elem2.reshape((num_snps, 1)), elem2.reshape((1, num_snps)))
    mat3 = np.dot(elem3.reshape((num_snps, 1)), elem3.reshape((1, num_snps)))
    w = mat1 * mat2 / mat3
    SBB = np.sum(np.triu(w*Y, k=1)) / np.sqrt(np.sum(np.triu(w ** 2, k=1)))
    return SBB

def get_weights(phenos):
    percentiles = (rankdata(phenos) - 1) / len(phenos)
    weights = -np.log(1 - percentiles)
    return weights / np.sum(weights)

def corr(x, w):
    """Weighted Correlation"""
    c = np.cov(x, aweights=w)
    d = np.diag(np.diag(c) ** -0.5)
    return np.dot(np.dot(d, c), d)

def continuous_buhmbox(genos, phenos, clist,snp_props):
    """
    cases, controls: Numpy array where each row is an indiv and each col is a snp
    clist: tuple of indices that are the snps for DB
    snp_props: 
    """
    num_snps = len(clist)
    snp_indivs = genos[:,clist]
    num_indivs = snp_indivs.shape[0]
    percent_holdout = 0.05
    num_holdout = int(0.05 * num_indivs)
    num_used_indivs = num_indivs - num_holdout
    
    # split into first 5% and rest
    perm = np.random.permutation(num_indivs)
    snp_indivs, phenos = snp_indivs[perm], phenos[perm]
    snp_indivs_holdout, snp_indivs = snp_indivs[:num_holdout], snp_indivs[num_holdout:]
    phenos_holdout, phenos = phenos[:num_holdout], phenos[num_holdout:]
    
    # find pi/gamma
    weights = get_weights(phenos_holdout)
    pi_plus = np.sum(snp_indivs_holdout * weights.reshape((num_holdout, 1)), axis=0) / 2
    pi_minus = np.sum(snp_indivs_holdout, axis=0) / (2*float(num_holdout))
    gamma = pi_plus/(1-pi_plus) / (pi_minus/(1-pi_minus))
    
    # get weights for not holdout sample and do BB
    weights = get_weights(phenos)
    n = float(num_snps)
    N = float(num_used_indivs)
    w2 = np.sum(weights ** 2)
    R = corr(snp_indivs.T, weights)
    Rp = np.corrcoef(snp_indivs.T)
    Y = (w2 - 1/N)**-0.5 * (R-Rp)
    
    # calculate SBB
    elem1 = np.sqrt(pi_minus*(1-pi_minus))
    elem2 = gamma-1
    elem3 = elem2 * pi_minus + 1
    mat1 = np.sqrt(np.dot(elem1.reshape((num_snps, 1)), elem1.reshape((1, num_snps))))
    mat2 = np.dot(elem2.reshape((num_snps, 1)), elem2.reshape((1, num_snps)))
    mat3 = np.dot(elem3.reshape((num_snps, 1)), elem3.reshape((1, num_snps)))
    w = mat1 * mat2 / mat3
    SBB = np.sum(np.triu(w*Y, k=1)) / np.sqrt(np.sum(np.triu(w ** 2, k=1)))
    #SBB = np.sum(np.triu(Y, k=1)) / ((n * (n-1)) / 2) ** 0.5
    return SBB

def get_mats_from_pop(pop, phen, z_thresh):
    genos, phenos, _ = pop
    mu = np.mean(phenos[:, phen])
    sigma = np.std(phenos[:, phen])
    
    cases_indices = np.where(phenos[:, phen] > (mu + z_thresh * sigma))
    controls_indices = np.where(phenos[:, phen] <= (mu + z_thresh * sigma))
    cases = genos[cases_indices]
    controls = genos[controls_indices]
#    return cases, controls
    return cases, genos # return full pop as controls
    
def get_clist(snps, phens):
    _, snp_betas = snps
    return tuple(set(np.where(snp_betas[:, phens] > 0)[0]))

def run_buhmbox_on_pop(pop, snps, snp_phens=0, case_phen=1, z=1.5):
    cases, controls = get_mats_from_pop(pop, case_phen, z)
    num_cases, _ = cases.shape
    controls_sub = controls[:num_cases, :]
    clist = get_clist(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return buhmbox(cases, controls, clist, snps)

def run_cont_buhmbox_on_pop(pop, snps, snp_phens=0, case_phen=1):
    genos, phenos, _ = pop
    clist = get_clist(snps, snp_phens)
    #print clist
    #print_snp_props([independent_snps[i] for i in clist])
    #print len(clist)
    return continuous_buhmbox(genos, phenos[:, case_phen], clist, snps)

def simulate_mult_h(file_path, runs=1, num_snps=100, num_inds=100000, h_vals=(1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05), bzs=[-0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5]):
    start = time.time()
    independent_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps], [num_snps, 0]]))
    pleiotropy_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps/2], [num_snps/2, num_snps/2]]))
    b = np.zeros(8).reshape((2, 2, 2))
    b[1, 0, 0] = num_snps/4
    b[0, 0, 1] = num_snps/4
    b[1, 0, 1] = num_snps*3/4
    b[0, 1, 0] = num_snps
    hetero_snps = generate_pss_model_generalized(num_phenos=3, num_snps=b)
    
    points = {"ic": [], "pc": [], "hc": []}
    for bz in bzs:
        points["ib-" + str(bz)] = []
        points["pb-" + str(bz)] = []
        points["hb-" + str(bz)] = []
    for h in h_vals:
        ic = []
        pc = []
        hc = []
        ibs = {bz: [] for bz in bzs}
        pbs = {bz: [] for bz in bzs}
        hbs = {bz: [] for bz in bzs}
        for i in range(0, runs):    
            print str(h) + "-" + str(i)
            independent_pop = generate_population(independent_snps, num_inds=num_inds, h=h)
            pleiotropic_pop = generate_population(pleiotropy_snps, num_inds=num_inds, h=h)
            hetero_pop = generate_hetero_population(hetero_snps, num_inds=num_inds, h=h)
            
            '''
            plot_inds(independent_pop)
            plot_inds(pleiotropic_pop)
            plot_inds(hetero_pop)
            plot_inds_alts(hetero_pop)
            '''

            ic.append(run_cont_buhmbox_on_pop(independent_pop, independent_snps))
            pc.append(run_cont_buhmbox_on_pop(pleiotropic_pop, pleiotropy_snps))
            hc.append(run_cont_buhmbox_on_pop(hetero_pop, hetero_snps))
            for bz in bzs:
                ibs[bz].append(run_buhmbox_on_pop(independent_pop, independent_snps, z=bz))
                pbs[bz].append(run_buhmbox_on_pop(pleiotropic_pop, pleiotropy_snps, z=bz))
                hbs[bz].append(run_buhmbox_on_pop(hetero_pop, hetero_snps, z=bz))
        for arr, name in ((ic, "ic"), (pc, "pc"), (hc, "hc")):
            points[name].append((h, np.mean(arr), np.std(arr)))
        for bz in bzs:
            for root, array_master in (("ib-", ibs), ("pb-", pbs), ("hb-", hbs)):
                name = root + str(bz)
                arr = array_master[bz]
                points[name].append((h, np.mean(arr), np.std(arr)))
        print str(h) + " is done"
    with open(file_path, "wb") as f:
        pickle.dump(points, f)
    print(time.time()-start)

def simulate_mult_n(file_path, runs=1, num_snps=100, h=0.3, num_inds_vals=(1000, 5000, 10000, 20000, 50000, 100000), bzs=[-0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5]):
    start = time.time()
    independent_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps], [num_snps, 0]]))
    pleiotropy_snps = generate_pss_model_generalized(num_phenos=2, num_snps=np.array([[0, num_snps/2], [num_snps/2, num_snps/2]]))
    b = np.zeros(8).reshape((2, 2, 2))
    b[1, 0, 0] = num_snps/4
    b[0, 0, 1] = num_snps/4
    b[1, 0, 1] = num_snps*3/4
    b[0, 1, 0] = num_snps
    hetero_snps = generate_pss_model_generalized(num_phenos=3, num_snps=b)
    
    points = {"ic": [], "pc": [], "hc": []}
    for bz in bzs:
        points["ib-" + str(bz)] = []
        points["pb-" + str(bz)] = []
        points["hb-" + str(bz)] = []
    for num_inds in num_inds_vals:
        ic = []
        pc = []
        hc = []
        ibs = {bz: [] for bz in bzs}
        pbs = {bz: [] for bz in bzs}
        hbs = {bz: [] for bz in bzs}
        for i in range(0, runs):    
            print str(num_inds) + "-" + str(i)
            independent_pop = generate_population(independent_snps, num_inds=num_inds, h=h)
            pleiotropic_pop = generate_population(pleiotropy_snps, num_inds=num_inds, h=h)
            hetero_pop = generate_hetero_population(hetero_snps, num_inds=num_inds, h=h)
            
            '''
            plot_inds(independent_pop)
            plot_inds(pleiotropic_pop)
            plot_inds(hetero_pop)
            plot_inds_alts(hetero_pop)
            '''

            ic.append(run_cont_buhmbox_on_pop(independent_pop, independent_snps))
            pc.append(run_cont_buhmbox_on_pop(pleiotropic_pop, pleiotropy_snps))
            hc.append(run_cont_buhmbox_on_pop(hetero_pop, hetero_snps))
            for bz in bzs:
                ibs[bz].append(run_buhmbox_on_pop(independent_pop, independent_snps, z=bz))
                pbs[bz].append(run_buhmbox_on_pop(pleiotropic_pop, pleiotropy_snps, z=bz))
                hbs[bz].append(run_buhmbox_on_pop(hetero_pop, hetero_snps, z=bz))
        for arr, name in ((ic, "ic"), (pc, "pc"), (hc, "hc")):
            points[name].append((num_inds, np.mean(arr), np.std(arr)))
        for bz in bzs:
            for root, array_master in (("ib-", ibs), ("pb-", pbs), ("hb-", hbs)):
                name = root + str(bz)
                arr = array_master[bz]
                points[name].append((num_inds, np.mean(arr), np.std(arr)))
        print str(num_inds) + " is done"
    with open(file_path, "wb") as f:
        pickle.dump(points, f)
    print(time.time()-start)

def print_info(file_path, param_name="h"):
    with open(file_path, "rb") as f:
        points = pickle.load(f)
    for name in sorted(points.keys()):
        print name + ":"
        for point in points[name]:
            print "  {}={}: mean={}, sd={}".format(param_name, *point)

def plot_info(file_path, xlabel="heritability"):
    with open(file_path, "rb") as f:
        points = pickle.load(f)
    
    things = {}
    for name in points:
        if name[1] == 'c':
            things[name] = 'c'
        else:
            things[name] = 'b-' + name.split('-')[1]
    vals = set(things.values())
    #point_colors = {'b-2.1': 'blue', 'b-1.5': 'green', 'b-1.8': 'purple', 'c': 'orange', 'b-1.2': 'red'}
        
    print "independent means:"
    for name in points:
        if name[0] == "i":
            plt.plot(*(zip(*points[name])[:2]), label=name)#, c=point_colors[things[name]])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB mean score")
    plt.title("Independent population - BB means")
    axes = plt.gca()
    axes.set_ylim([-1,1])
    plt.show()
    
    print "independent stds:"
    for name in points:
        if name[0] == "i":
            plt.plot(*(zip(*points[name])[::2]), label=name)#, c=point_colors[things[name]])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB score std")
    plt.title("Independent population - BB stds")
    axes = plt.gca()
    axes.set_ylim([0,2])
    plt.show()

    print "pleiotropic means:"
    for name in points:
        if name[0] == "p":
            plt.plot(*(zip(*points[name])[:2]), label=name)#, c=point_colors[things[name]])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB mean score")
    plt.title("Pleiotropic population - BB means")
    plt.show()
    
    print "pleiotropic stds:"
    for name in points:
        if name[0] == "p":
            plt.plot(*(zip(*points[name])[::2]), label=name)#, c=point_colors[things[name]])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB score std")
    plt.title("Pleiotropic population - BB stds")
    plt.show()
    
    print "heterogeneous means:"
    for name in points:
        if name[0] == "h":
            plt.plot(*(zip(*points[name])[:2]), label=name)#, c=point_colors[things[name]])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB mean score")
    plt.title("Heterogeneous population - BB means")
    plt.show()
    
    print "heterogeneous stds:"
    for name in points:
        if name[0] == "h":
            plt.plot(*(zip(*points[name])[::2]), label=name)#, c=point_colors[things[name]])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB score std")
    plt.title("Heterogeneous population - BB stds")
    plt.show()

def plot_info_final(file_path, xlabel="heritability"):
    with open(file_path, "rb") as f:
        points = pickle.load(f)
    
    del_names = []
    for name in points:
        split = name.split('-', 1)
        # all : ["1.5", "1.2", "0.9", "0.6", "0.3", "0.1", "0", "-0.1", "-0.3", "-0.6"]
        if split[0][1] != 'c' and name.split('-', 1)[1] not in ["1.5", "1.2", "0.9", "0.6", "0.3", "0", "-0.3", "-0.6"]:
            del_names.append(name)
    for name in del_names:
        del points[name]
    
    def get_sort_val(x):
        total = 0
        vals = {'i': 1, 'p':2, 'h': 3}
        total += vals[x[0]] * 100
        if x[1] != 'c':
            zval = int(float(x.split('-', 1)[1])*10)
            return total + zval
        else:
            return total + 99
    
    def sort_func(x, y):
        return get_sort_val(y) - get_sort_val(x)
    
    sorted_points = sorted(points.keys(),sort_func)
    
    zvals = set()
    
    labels = {}
    for name in points:
        if name[1] == 'c':
            labels[name] = 'Continuous BB'
        else:
            zval = name.split('-', 1)[1]
            zvals.add(float(zval))
            labels[name] = 'Naive BB cutoff ' + str(zval)
    
    colors = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(zvals), vmax=max(zvals)), cmap='cool')
    
    point_colors = {}
    for name in points:
        if name[1] == 'c':
            point_colors[name] = 'black'
        else:
            zval = float(name.split('-', 1)[1])
            point_colors[name] = colors.to_rgba(zval)

    #point_colors = {'b-2.1': 'blue', 'b-1.5': 'green', 'b-1.8': 'purple', 'c': 'orange', 'b-1.2': 'red'}
    print "independent means:"
    for name in sorted_points:
        if name[0] == "i":
            plt.plot(*(zip(*points[name])[:2]), label=labels[name], c=point_colors[name])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB mean score")
    plt.title("Independent population - BB means")
    axes = plt.gca()
    axes.set_ylim([-1,1])
    plt.show()
    
    print "independent stds:"
    for name in sorted_points:
        if name[0] == "i":
            plt.plot(*(zip(*points[name])[::2]), label=labels[name], c=point_colors[name])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB score std")
    plt.title("Independent population - BB stds")
    axes = plt.gca()
    axes.set_ylim([0,2])
    plt.show()

    print "pleiotropic means:"
    for name in sorted_points:
        if name[0] == "p":
            heritabilities, scores, stds = zip(*points[name])
            plt.errorbar(heritabilities, scores, yerr=stds, label=labels[name], c=point_colors[name])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB mean score")
    plt.title("Pleiotropic population - BB means")
    plt.show()
    
    print "heterogeneous means:"
    for name in sorted_points:
        if name[0] == "h":
            heritabilities, scores, stds = zip(*points[name])
            plt.errorbar(heritabilities, scores, yerr=stds, label=labels[name], c=point_colors[name])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("BB mean score")
    plt.title("Heterogeneous population - BB means")
    plt.show()

FILE_PATH_H = "data_heritabilities.pickle"
FILE_PATH_N = "info_num_inds.pickle"

if __name__=="__main__":
    if not os.path.exists(FILE_PATH_H):
        simulate_mult_h(FILE_PATH_H, runs=100, num_snps=100, num_inds=100000, h_vals=(1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05))
    if not os.path.exists(FILE_PATH_N):
        simulate_mult_n(FILE_PATH_N, runs=100, num_snps=100, h=0.5, num_inds_vals=(1000, 5000, 10000, 20000, 50000, 100000))
    #print_info(FILE_PATH_H, param_name="h")
    #print_info(FILE_PATH_N, param_name="n")
    plot_info_final(FILE_PATH_H, xlabel="heritabilities")
    plot_info_final(FILE_PATH_N, xlabel="number of individuals")
    
