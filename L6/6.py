import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

def laborator6():

    x = np.array([82, 106, 120, 68, 83, 89, 130, 92, 99, 89])

    # 1
    mu_init, sigma_init = 90, 10
    t = np.linspace(x.min()-20, x.max()+20, 200)
    pdf_init = norm.pdf(t, mu_init, sigma_init)
    plt.plot(t, pdf_init, label="N(90,10^2)")
    plt.legend()
    plt.show()

    # 2
    x1 = 82
    pdf_val_formula = 1/(np.sqrt(2*np.pi)*sigma_init)*np.exp(-(x1-mu_init)**2/(2*sigma_init**2))
    pdf_val_scipy = norm.pdf(x1, mu_init, sigma_init)
    print("f(82) formula:", pdf_val_formula)
    print("f(82) scipy  :", pdf_val_scipy)

    # 3
    pdf_prod = np.prod(norm.pdf(x, mu_init, sigma_init))
    print("Produs pdf (mu=90,sigma=10):", pdf_prod)

    # 4-5
    def prior_mu(m): return norm.pdf(m, 100, 50)
    def prior_sigma(s): return uniform.pdf(s, loc=1, scale=69)
    def likelihood(m, s): return np.prod(norm.pdf(x, m, s))
    def posterior(m, s): return likelihood(m, s)*prior_mu(m)*prior_sigma(s)

    print("Posterior (90,10):", posterior(90, 10))

    # 6
    mus = [70, 75, 80, 85, 90, 95, 100]
    sigmas = [5, 10, 15, 20]
    best_val = -1
    best_mu, best_sigma = None, None
    for m in mus:
        for s in sigmas:
            val = posterior(m, s)
            if val > best_val:
                best_val = val
                best_mu, best_sigma = m, s
    print("Cel mai bun model:", best_mu, best_sigma, best_val)

def main():
    laborator6()

if __name__ == "__main__":
    main()