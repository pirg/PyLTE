from astropy.io import ascii
import numpy as np

from astroquery.splatalogue import Splatalogue
from astropy import units as u
from astropy.table import Column
import astropy 

class LTE(object): 
  """Computes LTE """
  def __init__(self, species, part_name, idx_obs, points_per_line=100,extent=6,CDMS_file=None,Qfactor=1.,Dipole_factor=1.):
    self.part_name = part_name
    self.species = species
    self.idx_obs = idx_obs
    self.Qfactor = Qfactor
    self.parse_Q(self.part_name)
    if CDMS_file != None:
      self.linelist = self.from_CDMS(CDMS_file,Dipole_factor)
    else:
      self.linelist = self.trimmed_query(8000*u.MHz,50000*u.MHz,chemical_name=species)
    self.points_per_line = points_per_line
    self.extent = extent

  def from_CDMS(self,CDMS_file,Dipole_factor):
    data = ascii.read(CDMS_file)
    Q300 = self.compute_Q(300)
    cm2K = ((astropy.constants.h*astropy.constants.c/u.cm)/(astropy.constants.k_B*u.K)).decompose()
    data['EL_K'] = data['El_cm']*cm2K
    data['EU_K'] = (((data['Freq-MHz']*u.MHz)/astropy.constants.c+data['El_cm']/u.cm).to('1/cm')*cm2K).value
    data['Aij'] = Dipole_factor*pow(10,data['log10I'])*data['Freq-MHz']**2*Q300/data['g_u']*2.7964e-16/(np.exp(-data['EL_K']/300.)*(1-np.exp(-astropy.constants.h*data['Freq-MHz']*u.MHz/(astropy.constants.k_B*300.*u.K))))
    print data['Freq-MHz']
    print np.log10(data['Aij'])
    return data

  def trimmed_query(self, *args,**kwargs):
          S = Splatalogue(energy_max=500,
             energy_type='eu_k',energy_levels=['el4'],
             line_strengths=['ls4'],
             only_NRAO_recommended=True,
             show_upper_degeneracy=True)
          columns = ['Species','Chemical Name','Resolved QNs','Freq-GHz',
                     'Log<sub>10</sub> (A<sub>ij</sub>)',
                     'E_U (K)','Upper State Degeneracy' ]
          table = S.query_lines(*args, **kwargs)
          table.rename_column('Log<sub>10</sub> (A<sub>ij</sub>)','log10(Aij)')
          table['Aij'] = pow(10,table['log10(Aij)'])
          #table.remove_column('log10(Aij)')
          table.rename_column('E_U (K)','EU_K')
          table.rename_column('Resolved QNs','QNs')
          table.rename_column('Upper State Degeneracy','g_u')
          table['Freq-MHz'] = table['Freq-GHz']*1000.
          table.remove_column('Freq-GHz')        
          table.sort('Freq-MHz')
          self.remove_hfs(table)
          if self.idx_obs:
            table = table[self.idx_obs]
          return table

  def remove_hfs(self,table):
    """docstring for remove_hfs"""
    not_hfs_idx = []
    for i,row in enumerate(table):
      if "F"  not in row['QNs']:
        not_hfs_idx.append(i)
    table.remove_rows(not_hfs_idx)
  
  def parse_Q(self,species):
    colwidths = [6,13,7]+9*[7]
    self.partfunc = ascii.read("partfunc.txt", guess=False, format='fixed_width_no_header', col_starts=tuple([0])+tuple(np.cumsum(colwidths[0:-1])), col_ends=tuple(np.cumsum(colwidths)), names=('tag', 'molecule', 'nline',  '300K', '225K', '150K', '75K', '37.5K', '18.75K', '9.375 K', '5.0 K', '2.725 K'))
    row = self.partfunc[self.partfunc['molecule'] == species]
    row['0 K'] = row['9.375 K']
    self.qval = list(row.filled()[0].data)[3:]
    self.qval = np.array(map(float,self.qval))
    print self.qval
        
    
  def compute_Q(self, T):
      qval = self.qval
      temps = np.array([300, 225, 150, 75, 37.5, 18.75,9.375,5.0,2.725,0])
      idx = temps.argsort()
      temps = temps[idx]
      qval = qval[idx]
      qval = np.interp(T, temps, qval)      
      return pow(10,qval)*self.Qfactor
  
  def phi(self, Dv, freq, npoint):
    import scipy.stats.distributions as stats
    sigma = freq*u.MHz / (astropy.constants.c * np.sqrt(8 * np.log(2))) \
        * Dv*u.km/u.s  # Hz
    sigma = sigma.decompose()
    sigma_kms = (sigma*astropy.constants.c/(freq*u.MHz)).to('km/s')
    x = np.linspace(-self.extent*sigma,self.extent*sigma,npoint)
    x_kms = (x*astropy.constants.c/(freq*u.MHz)).to('km/s')
    y = stats.norm.pdf(x, 0., sigma)
    return x.to('MHz'), y*u.s, x_kms
  
  
  def J(self, T, freq):
      """
      Returns the radiation temperature

      This function returns the radiation temperature corresponding to
      the given kinetic temperature at the given frequency. See Eq. 1.28
      in the Tools of Radio Astronony by Rohlfs & Wilson.

      Arguments:
      T    -- kinetic temperature, in K
      freq -- frequency, in MHz

      """

      J = astropy.constants.h * freq / astropy.constants.k_B \
              / (np.exp((astropy.constants.h * freq) \
                                 / (astropy.constants.k_B * T * u.K)) - 1)

      return J
     
  def tau(self, freq, Aij, gu, Eu, Q, Ntot, Tex, Dv):
      Tbg = 2.73
      phi_vec = self.phi(Dv,freq, self.points_per_line)
      tau = astropy.constants.c**2 / (8 * np.pi * (freq)**2*u.MHz*u.MHz) * Aij*u.Hz \
          * Ntot/u.cm**2 * gu \
          * np.exp(-Eu / Tex) \
          / Q * (np.exp(astropy.constants.h * (freq)*u.MHz / (Tex*u.K * astropy.constants.k_B))-1)
      tau = tau * phi_vec[1]
      tau = tau.decompose()
      tb = (self.J(Tex, freq*u.MHz) - self.J(Tbg, freq*u.MHz)) * (1 - np.exp(-tau))
      tb = tb.decompose()
      Vlsr_vec = phi_vec[2]
      W = tb.sum()*(Vlsr_vec[1]-Vlsr_vec[0])
      return tau.max().value, freq*u.MHz+phi_vec[0], tb, W.to('K km/s').value, Vlsr_vec, freq*u.MHz, tau, phi_vec
  
    
  def intens_tau(self, Ntot, Tex, Dv):
    model = np.zeros((len(self.idx_obs),2))
    for i,idx in enumerate(self.idx_obs):
      # idx = idx[0].split(',')
      # idx = map(int,idx)
      for row in self.linelist[idx]:
        model[i,:] +=  self.tau(np.array(row['Freq-MHz']), np.array(row['Aij']), np.array(row['g_u']), np.array(row['EU_K']), self.compute_Q(Tex),  Ntot, Tex, Dv )[0:4:3]
    return model
    
  def line_profile(self, Ntot, Tex, Dv):
    return [self.tau(np.array(row['Freq-MHz']), np.array(row['Aij']), np.array(row['g_u']), np.array(row['EU_K']), self.compute_Q(Tex), Ntot, Tex, Dv )[2].value for row in self.linelist]

  def freq_vec(self, Ntot, Tex, Dv):
    return [self.tau(np.array(row['Freq-MHz']), np.array(row['Aij']), np.array(row['g_u']), np.array(row['EU_K']), self.compute_Q(Tex), Ntot, Tex, Dv )[5].value for row in self.linelist]

  def Vlsr_vec(self, Ntot, Tex, Dv):
    return [self.tau(np.array(row['Freq-MHz']), np.array(row['Aij']), np.array(row['g_u']), np.array(row['EU_K']), self.compute_Q(Tex), Ntot, Tex, Dv )[4].value for row in self.linelist]


