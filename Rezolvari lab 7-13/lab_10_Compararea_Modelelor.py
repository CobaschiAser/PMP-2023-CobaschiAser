import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt


def get_data(order, x_1, y_1):
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    return x_1s, y_1s


# 1

dummy_data = np.loadtxt("dummy.csv")
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
x_1s, y_1s = get_data(5, x_1, y_1)
order = 5
model_p1 = pm.Model()
model_p2 = pm.Model()
model_p3 = pm.Model()

# a)
with model_p1:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=order)
    epsilon = pm.HalfNormal("epsilon", 5)
    miu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
    idata_p1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

# b.1) sd = 100
with model_p2:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=100, shape=order)
    epsilon = pm.HalfNormal("epsilon", 5)
    miu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
    idata_p2 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

# b.2) sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
with model_p3:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    epsilon = pm.HalfNormal("epsilon", 5)
    miu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
    idata_p3 = pm.sample(2000, return_inferencedata=True)

# reprezentarea grafica a modelelor
idx = np.argsort(x_1s[0])

alpha_p1_post = idata_p1.posterior["alpha"].mean(("chain", "draw")).values
beta_p1_post = idata_p1.posterior["beta"].mean(("chain", "draw")).values
y_p1_post = alpha_p1_post + np.dot(beta_p1_post, x_1s)

alpha_p2_post = idata_p2.posterior["alpha"].mean(("chain", "draw")).values
beta_p2_post = idata_p2.posterior["beta"].mean(("chain", "draw")).values
y_p2_post = alpha_p2_post + np.dot(beta_p2_post, x_1s)

alpha_p3_post = idata_p3.posterior["alpha"].mean(("chain", "draw")).values
beta_p3_post = idata_p3.posterior["beta"].mean(("chain", "draw")).values
y_p3_post = alpha_p3_post + np.dot(beta_p3_post, x_1s)

plt.plot(x_1s[0][idx], y_p1_post[idx], "C1", label=f"order={order}, beta sd=10")
plt.plot(x_1s[0][idx], y_p2_post[idx], "C2", label=f"order={order}, beta sd=100")
plt.plot(x_1s[0][idx], y_p3_post[idx], "C3", label=f"order={order}, beta sd=(10, 0.1, 0.1, 0.1, 0.1)")

plt.scatter(x_1s[0], y_1s, c="C0", marker=".")
plt.legend()
plt.show()


# 2
# adăugăm alte date pentru a ajunge la 500, la o scară asemănătoare:
x_1_add = np.random.normal(np.mean(x_1),np.std(x_1),size=500-len(x_1))
y_1_add = np.random.normal(np.mean(y_1),np.std(y_1),size=500-len(y_1))
x_1_500 = np.concatenate((x_1,x_1_add))
y_1_500 = np.concatenate((y_1,y_1_add))

# vizualizarea datelor:
plt.scatter(x_1,y_1)
plt.scatter(x_1_add,y_1_add,color='m',alpha=0.5)

x_1s_500, y_1s_500 = get_data(5, x_1_500, y_1_500)
order = 5
model_p1_500 = pm.Model()
model_p2_500 = pm.Model()
model_p3_500 = pm.Model()

# a)
with model_p1_500:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=order)
    epsilon = pm.HalfNormal("epsilon", 5)
    miu = alpha + pm.math.dot(beta, x_1s_500)
    y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s_500)
    idata_p1_500 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

# b.1)
with model_p2_500:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=100, shape=order)
    epsilon = pm.HalfNormal("epsilon", 5)
    miu = alpha + pm.math.dot(beta, x_1s_500)
    y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s_500)
    idata_p2_500 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

# b.2)
with model_p3_500:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    epsilon = pm.HalfNormal("epsilon", 5)
    miu = alpha + pm.math.dot(beta, x_1s_500)
    y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s_500)
    idata_p3_500 = pm.sample(2000, return_inferencedata=True)

#reprezentarea grafica a modelelor
idx = np.argsort(x_1s_500[0])

alpha_p1_post_500 = idata_p1_500.posterior["alpha"].mean(("chain", "draw")).values
beta_p1_post_500 = idata_p1_500.posterior["beta"].mean(("chain", "draw")).values
y_p1_post_500 = alpha_p1_post_500 + np.dot(beta_p1_post_500, x_1s_500)

alpha_p2_post_500 = idata_p2_500.posterior["alpha"].mean(("chain", "draw")).values
beta_p2_post_500 = idata_p2_500.posterior["beta"].mean(("chain", "draw")).values
y_p2_post_500 = alpha_p2_post_500 + np.dot(beta_p2_post_500, x_1s_500)

alpha_p3_post_500 = idata_p3_500.posterior["alpha"].mean(("chain", "draw")).values
beta_p3_post_500 = idata_p3_500.posterior["beta"].mean(("chain", "draw")).values
y_p3_post_500 = alpha_p3_post_500 + np.dot(beta_p3_post_500, x_1s_500)

plt.plot(x_1s_500[0][idx], y_p1_post_500[idx], "C1", label=f"order={order}, beta sd=10")
plt.plot(x_1s_500[0][idx], y_p2_post_500[idx], "C2", label=f"order={order}, beta sd=100")
plt.plot(x_1s_500[0][idx], y_p3_post_500[idx], "C3", label=f"order={order}, beta sd=(10, 0.1, 0.1, 0.1, 0.1)")

plt.scatter(x_1s_500[0], y_1s_500, c="C0", marker=".")
plt.legend()
plt.show()


# 3
# modelul liniar (din curs):
x_1s, y_1s = get_data(1, x_1, y_1)
with pm.Model() as model_l:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + beta * x_1s[0]
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True)

model_p_ord2 = pm.Model()
model_p_ord3 = pm.Model()

# modelul patratic
order = 2
x_1s, y_1s = get_data(order, x_1, y_1)
with model_p_ord2:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=order)
    epsilon = pm.HalfNormal("epsilon", 5)
    miu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
    idata_p_ord2 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)


# modelul cubic
order = 3
x_1s, y_1s = get_data(order, x_1, y_1)
with model_p_ord3:
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=order)
    epsilon = pm.HalfNormal("epsilon", 5)
    miu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal("y_pred", mu=miu, sigma=epsilon, observed=y_1s)
    idata_p_ord3 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

pm.compute_log_likelihood(idata_l, model=model_l)
pm.compute_log_likelihood(idata_p_ord2, model=model_p_ord2)
pm.compute_log_likelihood(idata_p_ord3, model=model_p_ord3)

# calc waic si compararea
cmp_waic = az.compare({'model_l':idata_l, 'model_p_ord2':idata_p_ord2, 'model_p_ord3':idata_p_ord3},method='BB-pseudo-BMA', ic="waic", scale="deviance")
print(cmp_waic)
az.plot_compare(cmp_waic)

# calc loo si compararea
cmp_loo = az.compare({'model_l':idata_l, 'model_p_ord2':idata_p_ord2, 'model_p_ord3':idata_p_ord3},method='BB-pseudo-BMA', ic="loo", scale="deviance")
print(cmp_loo)
az.plot_compare(cmp_loo)