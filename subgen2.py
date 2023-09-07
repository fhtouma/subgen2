''' generate Monte-Carlo samples of subhaloes.
'''
import numpy as np
import matplotlib.pyplot as plt
from nfw import NFWHalo,mean_concentration
import emcee
import scipy.integrate
from scipy import special

def InfallMass2StellarMass(mAcc):
    '''mAcc: infall mass, in 1e10Msun/h
        output: stellar mass, in 1e10Msun/h
        reference: Wang, De Lucia, Weinmann 2013, MNRAS 431, for satellite galaxies.'''
    M0=5.23e1 #unit:1e10Msun/h
    k=10**0.30*0.73 #unit:1e10Msun/h
    alpha=0.298
    beta=1.99
  
    MvcInMvb=0.733 #correction from 200*rho_crit mass to 200*rho_background mass, which is close to bound mass
    m=mAcc*MvcInMvb/M0
  
    return 2*k/(m**-alpha+m**-beta)

def halfmodemass(WDMmass, cosmo=None):
    '''WDMmass: thermal relic warm dark matter particle mass, in keV
       cosmo: cosmology parameters, including [Omega_m, Omega_b, h], Planck15 as default
       output: halfmodemass, in 1e10Msun/h
       reference: Bode01, Lovell14.'''
    mu = 1.2
    G = 43007.1
    H = 0.1
    if (cosmo is not None):
        Om = cosmo[0]
        Ob = comso[1]
        Owdm = Om-Ob
        h = cosmo[2]
    else:
        Ob = 0.0491
        Om = 0.3156
        Owdm = Om - Ob
        h = 0.6727  
    alpha = 0.05 * pow(WDMmass,-1.15) * pow(Owdm/0.4, 0.15) * pow(h / 0.65, 1.3)
    rho_omega = Om * 3 * H**2 / (8 * np.pi * G) * 1e9
    khm = pow(pow(2, 1/((5/mu)) )-1,  1/(2*mu)) / alpha
    lambda_hm =  2 * np.pi / khm
    M_hm = (4 * np.pi / 3) * rho_omega * (lambda_hm / 2)**3
    return M_hm

class ModelParameter:
    ''' container for model parameters. contains fs, A, alpha, mustar, beta parameters.'''
    def __init__(self,M, WDMmass = None):
        ''' model parameters for a given host mass M, in units of 1e10Msun/h '''
        self.fs=0.58 #fraction of survived subhaloes
        self.A=0.11*M**-0.05 #infall mass function amplitude
        self.alpha=0.95 #infall mass function slope
        self.mustar=0.65*M**-0.03 #stripping function amplitude
        self.beta = 1.7*M**-0.04 #stripping function slope
        self.sigma_mu= 0.98 
        self.WDMmass = WDMmass
        
        if WDMmass: # WDM mass function ratio: (1+(kappa * Mhm / Mx)**eta)**gamma
            self.Mhm = halfmodemass(self.WDMmass)  #Calculated by WDMmass
            self.kappa=2.29
            self.eta=1
            self.gamma=-0.68
            self.beta *= (1 + self.Mhm**0.248)**0.267
            self.mustar *= (1 + self.Mhm**0.5476)**-0.31
            self.sigma_mu *= (1 + self.Mhm**0.44)**0.17

class SubhaloSample:
    ''' a sample of subhaloes '''
    def __init__(self,M,N=1e4,MMinInfall=1e-5,MMaxInfall=1e-1,Rmax=2,C=None,include_disruption=True,weighted_sample=False,WDMmass=None):
        ''' initialize the sampling parameters.
  
        M: host mass in 1e10Msun/h
        N: number of subhaloes to generate
        MMinInfall: minimum infall mass, in unit of host mass
        MMaxInfall: maximum infall mass, in unit of host mass
        Rmax: maximum radius to sample, in unit of host virial radius
        C: host concentration. If None, it will be determined by the mean mass-concentration relation.
        include_disruption: whether to include disrupted subhaloes in the sample.
        weighted_sample: whether to sample the mass in a weighted way, so that different mass ranges are sampled equally well. this is useful if you have a large dynamic range in mass, e.g., from 10^-6 to 10^12 Msun.
	WDMmass: thermal relic warm dark matter mass, in unit of keV, if None, it will generate a CDM subhalo sample. weighted_sample should be set as False if you want to generate a WDM subhalo sample.
        '''
        
        self.M=M
        self.WDMmass=WDMmass
        self.Host=NFWHalo(M,C) 
	#self.Host.density=my_density_func #to use custom density, overwrite Host.density() function.
        self.C=self.Host.C
        self.HOD=ModelParameter(self.M, self.WDMmass)

        self.Rmax=Rmax
        self.Msample=self.Host.mass(Rmax*self.Host.Rv) #mass inside Rmax

        self.mAccMin=MMinInfall*M
        self.mAccMax=MMaxInfall*M
        self.n=int(N) #sample size
        # expected  number of subhaloes per host for WDM numerical integration
        if WDMmass:
            #hmf = lambda x: pow(x,-a-1) * pow(1 + pow(k * xhm / x ,b), g)
            self.nPred=self.HOD.A*self.Msample* scipy.integrate.quad(lambda x: pow(x,-self.HOD.alpha-1) * pow(1 + pow(self.HOD.kappa * self.HOD.Mhm / x ,self.HOD.eta), self.HOD.gamma), self.mAccMin, self.mAccMax)[0]
        else:
            self.nPred=self.HOD.A*self.Msample*(self.mAccMin**-self.HOD.alpha-self.mAccMax**-self.HOD.alpha)/self.HOD.alpha #expected number of subhaloes per host.
        self.include_disruption=include_disruption
        if include_disruption:
            self.nSurvive=int(N*self.HOD.fs)
            self.nDisrupt=N-self.nSurvive
        else:
            self.nPred*=self.HOD.fs #only survived subhaloes are generated.
            self.nSurvive=N
            self.nDisrupt=int(N/self.HOD.fs*(1-self.HOD.fs))
        self.weighted_sample=weighted_sample
        
        
            

    def _lnPDF(self, x):
        ''' R in units of Rv'''
        lnmu,lnR=x
        lnmubar=np.log(self.HOD.mustar)+self.HOD.beta*lnR
        dlnmu=lnmu-lnmubar
        if lnmu>0: #mu<1
            return -np.inf
        if dlnmu>np.log(4.2): #mu<mumax=4.2*mubar
            return -np.inf
        if lnR>np.log(self.Rmax):
            return -np.inf
	
        lnPDFmu=-0.5*(dlnmu/self.HOD.sigma_mu)**2
        lnPDFR=3.*lnR+np.log(self.Host.density(np.exp(lnR)*self.Host.Rv)) #dM/dlnR=rho*R^3. 
        return lnPDFmu+lnPDFR

    def assign_mu_R(self, nwalkers=8, nburn=200, plot_chain=True):
        '''run emcee to sample mu and R '''
        nsteps=int(self.n/nwalkers+1+nburn) #one more step to make up for potential round-off in N/nwalkers
        print ('running %d steps'%nsteps)
        ndim=2
        x00=np.array([-0.5,-0.5])
        x0=np.kron(np.ones([nwalkers,1]),x00)#repmat to nwalkers rows
        x0+=(np.random.rand(ndim*nwalkers).reshape(nwalkers,ndim)-0.5)*0.1 #random offset, [-0.5,0.5]*0.1
        sampler=emcee.EnsembleSampler(nwalkers,ndim,self._lnPDF)
        sampler.run_mcmc(x0,nsteps)
        if plot_chain:
            plt.figure()
            labels=[r"$\ln \mu$",r"$\ln R/R_{200}$"]
            for i in range(ndim):
                plt.subplot(ndim,1,i+1)
                for j in range(nwalkers):
                    plt.plot(range(nsteps),sampler.chain[j,:,i],'.')
                plt.ylabel(labels[i])
        plt.plot([nburn,nburn],plt.ylim(),'k--')
        plt.xlabel('Step')
        plt.subplot(211)
        plt.title('%d walkers, %d burn-in steps assumed'%(nwalkers,nburn), fontsize=10)
	#==========extract mu and R===========
        sample=sampler.chain[:,nburn:,:]
        flatchain=sample.reshape([-1,ndim])[-self.n:] #take the last N entries
        flatchain=np.exp(flatchain) #from log() to linear
        self.mu,self.R=flatchain.T
#==========disruptions===============

                    
    def project_radius(self):
        phi=np.arccos(np.random.rand(self.n)*2-1.) #random angle around the z-axis
        self.Rp=self.R*np.sin(phi) #projected radius

    def assign_mass(self):
        '''sample m and mAcc'''
        
        if (self.WDMmass is None):
            if self.weighted_sample:
                lnmmin,lnmmax=np.log(self.mAccMin), np.log(self.mAccMax)
                lnmAcc=np.random.rand(self.n)*(lnmmax-lnmmin)+lnmmin #uniform distribution between lnmmin and lnmmax
                self.mAcc=np.exp(lnmAcc)
                self.weight=self.mAcc**-self.HOD.alpha
                self.weight=self.weight/self.weight.sum()*self.nPred #w/sum(w)*Npred, equals to dN/dlnm*[delta(lnm)/N] as N->inf
	  #self.weight=self.HOD.fs*self.HOD.A*Host.Msample*mAcc**-self.HOD.alpha*(lnmmax-lnmmin)/self.n #the weight is dN/dlnm*[delta(lnm)/N]
                print (np.sum(self.weight), self.nPred)
            else:
                mmax,mmin=self.mAccMin**(-self.HOD.alpha),self.mAccMax**(-self.HOD.alpha) 
                mAcc=np.random.rand(self.n)*(mmax-mmin)+mmin #uniform in m**-alpha which is proportional to N
                self.mAcc=mAcc**-(1./self.HOD.alpha)
                self.weight=1.*self.nPred/self.n*np.ones(self.n)
        else:
            naccpect = 0
            self.mAcc=np.zeros(self.n)
            mmax,mmin=self.mAccMin**(-self.HOD.alpha),self.mAccMax**(-self.HOD.alpha)
            A = 1/scipy.integrate.quad(lambda x: pow(x,-self.HOD.alpha-1) * pow(1 + pow(self.HOD.kappa * self.HOD.Mhm / x ,self.HOD.eta), self.HOD.gamma), self.mAccMin, self.mAccMax)[0]
            px_wdm=lambda x: A * pow(x,-self.HOD.alpha-1) * pow(1 + pow(self.HOD.kappa * self.HOD.Mhm / x ,self.HOD.eta), self.HOD.gamma) if (x>=self.mAccMin)&(x<=self.mAccMax) else 0
            px_cdm=lambda x: self.HOD.alpha * pow(x,-self.HOD.alpha-1) / (mmax - mmin)
            kwc = px_wdm(self.mAccMax) / px_cdm(self.mAccMax)
            while naccpect < self.n:
                u0 = np.random.rand()
                mi = np.random.rand()*(mmax-mmin)+mmin
                mi = mi**-(1/self.HOD.alpha)
                if u0 <= px_wdm(mi) / (kwc * px_cdm(mi)):
                    self.mAcc[naccpect]=mi
                    naccpect+=1
            self.weight=1.*self.nPred/self.n*np.ones(self.n)
        
        if self.include_disruption:
            if self.WDMmass is None:
                self.mu[self.nSurvive:]=0. #trailing masses set to 0.
            else:
                for i in range(self.n):
                    q = np.random.rand()
                    fsw = self.HOD.fs * (1 + 3.81 * self.M**-0.6 * ((self.mAcc[i] / self.M) / (self.HOD.Mhm / self.M )**0.5  )**-0.95 )**-0.85
                    if q > fsw:
                        self.mu[i] = 0
        
        self.m=self.mAcc*self.mu

    def populate(self, plot_chain=True):
        '''populate the sample with m,mAcc,R and weight'''
        self.assign_mu_R(plot_chain=plot_chain)
        self.assign_mass()
	
    def assign_stellarmass(self): 
        '''generate stellar mass from infall mass, according to an abundance matching model'''
        logMstar=np.log10(InfallMass2StellarMass(self.mAcc))
        sigmaLogMstar=0.192
        deltaLogMstar=np.random.normal(0, sigmaLogMstar, self.n)
        self.mStar=10**(logMstar+deltaLogMstar)
  
    def assign_annihilation_emission(self, concentration_model='Ludlow'):
        
        '''generate annihilation luminosity
        concentration_model: 'MaccioW1', Maccio08 relation with WMAP 1 cosmology
						 'Ludlow', Ludlow14 relation with WMAP 1 cosmology'''
	#first generate concentration from infall mass, according to a mass-concetration relation
        logC=np.log10(mean_concentration(self.mAcc[:self.nSurvive],concentration_model)) #mean
        sigmaLogC=0.13 
        deltaLogC=np.random.normal(0, sigmaLogC, self.nSurvive) #scatter
        cAcc=10**(logC+deltaLogC)
        SatHalo=[NFWHalo(self.mAcc[i], cAcc[i]) for i in range(self.nSurvive)]
        rt=np.array([SatHalo[i].radius(self.m[i]) for i in range(self.nSurvive)]) #truncation radius
        self.L=np.zeros_like(self.m)
        self.L[:self.nSurvive]=np.array([SatHalo[i].luminosity(rt[i]) for i in range(self.nSurvive)]) #truncated luminosity

    def save(self, outfile, save_all=False):
        ''' save the sample to outfile.
        if save_all=True, save all the properties; otherwise only R,m,mAcc,weight will be saved.'''
        if save_all:
            np.savetxt(outfile, np.array([self.R,self.m,self.mAcc,self.weight,self.Rp,self.mStar,self.L]).T, header='R/R200, m/[1e10Msun/h], mAcc/[1e10Msun/h], weight, Rp/R200, mStar/[1e10Msun/h], Luminosity/[(1e10Msun/h)^2/(kpc/h)^3]')
        else:
            np.savetxt(outfile, np.array([self.R,self.m,self.mAcc,self.weight]).T, header='R/R200, m/[1e10Msun/h], mAcc/[1e10Msun/h], weight')
