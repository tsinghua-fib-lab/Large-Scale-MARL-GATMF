import numpy as np
import time

class Model:
    def __init__(self,
                 num_seeds=1,
                 debug=False,
                 clip_poisson_approximation=True,
                 random_simulation=False):

        self.num_seeds = num_seeds
        self.debug = debug
        self.clip_poisson_approximation = clip_poisson_approximation
        self.random_simulation=random_simulation

    def init_exogenous_variables(self,
                                 poi_areas,
                                 cbg_sizes,
                                 p_sick_at_t0,
                                 poi_psi,
                                 home_beta,
                                 cbg_attack_rates_original,
                                 cbg_death_rates_original,
                                 poi_cbg_visits_list=None,
                                 poi_dwell_time_correction_factors=None,
                                 just_compute_r0=False,
                                 latency_period=96,  # 4 days
                                 infectious_period=84,  # 3.5 days
                                 confirmation_rate=.1,
                                 confirmation_lag=168,  # 7 days
                                 death_lag=432,  # 18 days
                                 no_print=False,
                                 ):
        self.M = len(poi_areas)
        self.N = len(cbg_sizes)
        self.MAX_T=len(poi_cbg_visits_list)

        self.PSI = poi_psi
        self.POI_AREAS = poi_areas
        self.DWELL_TIME_CORRECTION_FACTORS = poi_dwell_time_correction_factors
        self.POI_FACTORS = self.PSI / poi_areas
        if poi_dwell_time_correction_factors is not None:
            self.POI_FACTORS = poi_dwell_time_correction_factors * self.POI_FACTORS
            self.included_dwell_time_correction_factors = True
        else:
            self.included_dwell_time_correction_factors = False
        self.POI_CBG_VISITS_LIST = poi_cbg_visits_list
        self.clipping_monitor = {
        'num_base_infection_rates_clipped':[],
        'num_active_pois':[],
        'num_poi_infection_rates_clipped':[],
        'num_cbgs_active_at_pois':[],
        'num_cbgs_with_clipped_poi_cases':[]}

        self.CBG_SIZES = cbg_sizes
        self.HOME_BETA = home_beta
        self.CBG_ATTACK_RATES_ORIGINAL = cbg_attack_rates_original
        self.CBG_DEATH_RATES_ORIGINAL = cbg_death_rates_original
        self.LATENCY_PERIOD = latency_period
        self.INFECTIOUS_PERIOD = infectious_period
        self.P_SICK_AT_T0 = p_sick_at_t0

        self.VACCINATION_VECTOR = np.zeros((self.num_seeds,self.N),dtype=np.float32)
        self.VACCINE_ACCEPTANCE = np.ones((self.num_seeds,self.N),dtype=np.float32)
        self.PROTECTION_RATE = 1.0

        self.just_compute_r0 = just_compute_r0
        self.confirmation_rate = confirmation_rate
        self.confirmation_lag = confirmation_lag
        self.death_lag = death_lag

        self.CBG_ATTACK_RATES_NEW = self.CBG_ATTACK_RATES_ORIGINAL * (1-self.PROTECTION_RATE*self.VACCINATION_VECTOR/self.CBG_SIZES)
        self.CBG_DEATH_RATES_NEW = self.CBG_DEATH_RATES_ORIGINAL
        self.CBG_ATTACK_RATES_NEW = np.clip(self.CBG_ATTACK_RATES_NEW, 0, None)
        self.CBG_DEATH_RATES_NEW = np.clip(self.CBG_DEATH_RATES_NEW, 0, None)
        self.CBG_DEATH_RATES_NEW = np.clip(self.CBG_DEATH_RATES_NEW, None, 1)

        assert((self.CBG_DEATH_RATES_NEW>=0).all())
        assert((self.CBG_DEATH_RATES_NEW<=1).all())

    def init_endogenous_variables(self):

        self.P0 = np.random.binomial(self.CBG_SIZES,self.P_SICK_AT_T0,size=(self.num_seeds, self.N))

        self.cbg_latent = self.P0
        self.cbg_infected = np.zeros((self.num_seeds, self.N),dtype=np.float32)
        self.cbg_removed = np.zeros((self.num_seeds, self.N),dtype=np.float32)
        self.cases_to_confirm = np.zeros((self.num_seeds, self.N),dtype=np.float32)
        self.new_confirmed_cases = np.zeros((self.num_seeds, self.N),dtype=np.float32)
        self.deaths_to_happen = np.zeros((self.num_seeds, self.N),dtype=np.float32)
        self.new_deaths = np.zeros((self.num_seeds, self.N),dtype=np.float32)
        self.C2=np.zeros((self.num_seeds, self.N),dtype=np.float32)
        self.D2=np.zeros((self.num_seeds, self.N),dtype=np.float32)

        self.VACCINATION_VECTOR = np.zeros((self.num_seeds,self.N),dtype=np.float32)
        self.CBG_ATTACK_RATES_NEW = self.CBG_ATTACK_RATES_ORIGINAL * (1-self.PROTECTION_RATE*self.VACCINATION_VECTOR/self.CBG_SIZES)
        self.CBG_DEATH_RATES_NEW = self.CBG_DEATH_RATES_ORIGINAL
        self.CBG_ATTACK_RATES_NEW = np.clip(self.CBG_ATTACK_RATES_NEW, 0, None)
        self.CBG_DEATH_RATES_NEW = np.clip(self.CBG_DEATH_RATES_NEW, 0, None)
        self.CBG_DEATH_RATES_NEW = np.clip(self.CBG_DEATH_RATES_NEW, None, 1)

        self.L_1=[]
        self.I_1=[]
        self.R_1=[]
        self.C_1=[]
        self.D_1=[]
        self.T1=[]
        self.t = 0
        self.C=[0]
        self.D=[0]
        self.history_C2 = []
        self.history_D2 = []
        self.epidemic_over = False

    def reset_random_seed(self):
        np.random.seed(np.random.randint(10000,size=self.num_seeds))

    def add_vaccine(self,vaccine_vector):
        self.VACCINATION_VECTOR+=vaccine_vector
        self.VACCINATION_VECTOR = np.clip(self.VACCINATION_VECTOR, None, (self.CBG_SIZES*self.VACCINE_ACCEPTANCE))
        self.CBG_ATTACK_RATES_NEW = self.CBG_ATTACK_RATES_ORIGINAL * (1-self.PROTECTION_RATE*self.VACCINATION_VECTOR/self.CBG_SIZES)
        self.CBG_ATTACK_RATES_NEW = np.clip(self.CBG_ATTACK_RATES_NEW, 0, None)

    def get_new_infectious(self):
        if self.random_simulation:
            new_infectious = np.random.binomial(np.round(self.cbg_latent).astype(int), 1 / self.LATENCY_PERIOD)
        else:
            new_infectious = self.cbg_latent / self.LATENCY_PERIOD

        return new_infectious

    def get_new_removed(self):
        if self.random_simulation:
            new_removed = np.random.binomial(np.round(self.cbg_infected).astype(int), 1 / self.INFECTIOUS_PERIOD)
        else:
            new_removed=self.cbg_infected / self.INFECTIOUS_PERIOD

        return new_removed

    def format_floats(self, arr):
        return [int(round(x)) for x in arr]

    def simulate_disease_spread(self,length=24,verbosity=1,no_print=False):
        assert(self.t<self.MAX_T)
        t_start=self.t
        time_start=time.time()

        while self.t-t_start < length:
            iter_t0 = time.time()
            if (verbosity > 0) and (self.t % verbosity == 0):
                L = np.sum(self.cbg_latent, axis=1)
                I = np.sum(self.cbg_infected, axis=1)
                R = np.sum(self.cbg_removed, axis=1)

                self.T1.append(self.t)
                self.L_1.append(L)
                self.I_1.append(I)
                self.R_1.append(R)
                self.C_1.append(self.C)
                self.D_1.append(self.D)

                self.history_C2.append(self.C2)
                self.history_D2.append(self.D2)

                if(no_print==False):
                    print('t:',self.t,'L:',L,'I:',I,'R',R,'C',self.C,'D',self.D)

            self.update_states(self.t)
            C1 = np.sum(self.new_confirmed_cases,axis=1)
            self.C2=self.C2+self.new_confirmed_cases
            self.C[0]=self.C[0]+C1
            D1 = np.sum(self.new_deaths,axis=1)
            self.D2=self.D2+self.new_deaths
            self.D[0]=self.D[0]+D1
            if self.debug and verbosity > 0 and self.t % verbosity == 0:
                print('Num active POIs: %d. Num with infection rates clipped: %d' % (self.num_active_pois, self.num_poi_infection_rates_clipped))
                print('Num CBGs active at POIs: %d. Num with clipped num cases from POIs: %d' % (self.num_cbgs_active_at_pois, self.num_cbgs_with_clipped_poi_cases))
            if self.debug:
                print("Time for iteration %i: %2.3f seconds" % (self.t, time.time() - iter_t0))

            self.t += 1

    def empty_record(self):
        self.L_1=[]
        self.I_1=[]
        self.R_1=[]
        self.C_1=[]
        self.D_1=[]
        self.T1=[]
        self.history_C2 = []
        self.history_D2 = []

    def output_record(self,full=False):
        cbg_all_affected = self.cbg_latent + self.cbg_infected + self.cbg_removed
        total_affected = np.sum(cbg_all_affected, axis=1)

        if full:
            print('Output records')
            o_T1=np.array(self.T1,dtype=np.float32)
            o_L_1=np.array(self.L_1,dtype=np.float32)
            o_I_1=np.array(self.I_1,dtype=np.float32)
            o_R_1=np.array(self.R_1,dtype=np.float32)
            o_C2=np.array(self.C2,dtype=np.float32)
            o_D2=np.array(self.D2,dtype=np.float32)
            o_history_C2=np.array(self.history_C2,dtype=np.float32)
            o_history_D2=np.array(self.history_D2,dtype=np.float32)
            o_total_affected=np.array(total_affected,dtype=np.float32)
            o_cbg_all_affected=np.array(cbg_all_affected,dtype=np.float32)

            o_history_C2=np.transpose(o_history_C2,(1,0,2))
            o_history_D2=np.transpose(o_history_D2,(1,0,2))

            return o_T1,o_L_1,o_I_1,o_R_1,o_C2,o_D2, o_total_affected, o_history_C2, o_history_D2, o_cbg_all_affected

        else:

            o_history_C2=np.array(self.history_C2,dtype=np.float32)[-1,:,:]
            o_history_D2=np.array(self.history_D2,dtype=np.float32)[-1,:,:]

            return o_history_C2,o_history_D2

    def update_states(self, t):
        self.get_new_cases(t)
        new_infectious = self.get_new_infectious()
        new_removed = self.get_new_removed()
        if not self.just_compute_r0:
            self.cbg_latent = self.cbg_latent + self.cbg_new_cases - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed

            if self.random_simulation:
                self.new_confirmed_cases = np.random.binomial(np.round(self.cases_to_confirm).astype(int), 1/self.confirmation_lag)
                new_cases_to_confirm = np.random.binomial(np.round(new_infectious).astype(int), self.confirmation_rate)
            else:
                self.new_confirmed_cases=self.cases_to_confirm/self.confirmation_lag
                new_cases_to_confirm=new_infectious*self.confirmation_rate

            self.cases_to_confirm = self.cases_to_confirm + new_cases_to_confirm - self.new_confirmed_cases

            if self.random_simulation:
                self.new_deaths = np.random.binomial(np.round(self.deaths_to_happen).astype(int), 1/self.death_lag)
                new_deaths_to_happen = np.random.binomial(np.round(new_infectious).astype(int), self.CBG_DEATH_RATES_NEW)
            else:
                self.new_deaths=self.deaths_to_happen/self.death_lag
                new_deaths_to_happen=new_infectious*self.CBG_DEATH_RATES_NEW

            self.deaths_to_happen = self.deaths_to_happen + new_deaths_to_happen - self.new_deaths
        else:
            self.cbg_latent = self.cbg_latent - new_infectious
            self.cbg_infected = self.cbg_infected + new_infectious - new_removed
            self.cbg_removed = self.cbg_removed + new_removed + self.cbg_new_cases

    def get_new_cases(self, t):
        cbg_densities = self.cbg_infected / self.CBG_SIZES
        overall_densities = (np.sum(self.cbg_infected, axis=1) / np.sum(self.CBG_SIZES)).reshape(-1, 1)
        num_sus = np.clip(self.CBG_SIZES - self.cbg_latent - self.cbg_infected - self.cbg_removed, 0, None)
        sus_frac = num_sus / self.CBG_SIZES

        if self.PSI > 0:
            cbg_base_infection_rates = self.HOME_BETA * self.CBG_ATTACK_RATES_NEW * cbg_densities
            cbg_base_infection_rates=np.nan_to_num(cbg_base_infection_rates)
        else:
            cbg_base_infection_rates = np.tile(overall_densities, self.N) * self.HOME_BETA
        self.num_base_infection_rates_clipped = np.sum(cbg_base_infection_rates > 1)
        cbg_base_infection_rates = np.clip(cbg_base_infection_rates, None, 1.0)

        if self.POI_CBG_VISITS_LIST is not None:
            poi_cbg_visits = self.POI_CBG_VISITS_LIST[t]
            poi_visits = poi_cbg_visits @ np.ones(poi_cbg_visits.shape[1])

        if not self.just_compute_r0:
            self.num_active_pois = np.sum(poi_visits > 0)
            col_sums = np.squeeze(np.array(poi_cbg_visits.sum(axis=0)))
            self.cbg_num_out = col_sums
            poi_infection_rates = self.POI_FACTORS * (poi_cbg_visits @ cbg_densities.T).T
            self.num_poi_infection_rates_clipped = np.sum(poi_infection_rates > 1)
            if self.clip_poisson_approximation:
                poi_infection_rates = np.clip(poi_infection_rates, None, 1.0)

            cbg_mean_new_cases_from_poi = self.CBG_ATTACK_RATES_NEW * sus_frac * (poi_infection_rates @ poi_cbg_visits)
            cbg_mean_new_cases_from_poi=np.nan_to_num(cbg_mean_new_cases_from_poi)

            if self.random_simulation:
                num_cases_from_poi = np.random.poisson(cbg_mean_new_cases_from_poi)
            else:
                num_cases_from_poi=cbg_mean_new_cases_from_poi
            self.num_cbgs_active_at_pois = np.sum(cbg_mean_new_cases_from_poi > 0)

        self.num_cbgs_with_clipped_poi_cases = np.sum(num_cases_from_poi > num_sus)
        self.cbg_new_cases_from_poi = np.clip(num_cases_from_poi, None, num_sus)
        num_sus_remaining = num_sus - self.cbg_new_cases_from_poi

        if self.random_simulation:
            self.cbg_new_cases_from_base = np.random.binomial(np.round(num_sus_remaining).astype(int),cbg_base_infection_rates)
        else:
            self.cbg_new_cases_from_base = num_sus_remaining*cbg_base_infection_rates

        self.cbg_new_cases = self.cbg_new_cases_from_poi + self.cbg_new_cases_from_base

        self.clipping_monitor['num_base_infection_rates_clipped'].append(self.num_base_infection_rates_clipped)
        self.clipping_monitor['num_active_pois'].append(self.num_active_pois)
        self.clipping_monitor['num_poi_infection_rates_clipped'].append(self.num_poi_infection_rates_clipped)
        self.clipping_monitor['num_cbgs_active_at_pois'].append(self.num_cbgs_active_at_pois)
        self.clipping_monitor['num_cbgs_with_clipped_poi_cases'].append(self.num_cbgs_with_clipped_poi_cases)
        assert (self.cbg_new_cases <= num_sus).all()