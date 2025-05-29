"""
erbium_core.py
==============
Clase principal del simulador de iones de Erbio (Er³⁺) - Modelo de 3 niveles
"""

import numpy as np
from scipy.integrate import odeint

class ErbiumSimulator:
    """
    Simulador avanzado para iones de Erbio con modelo de 3 niveles energéticos
    Incluye análisis de tasa de emisión óptica
    """
    
    def __init__(self, N0=1e6, P_pump=300, tau21=10e-3, tau10=1e-6, tau20=50e-6):
        """
        Inicializa el simulador con parámetros físicos
        
        Parámetros:
        -----------
        N0 : float
            Número total de iones
        P_pump : float
            Potencia de bombeo (tasa de excitación 0→2)
        tau21 : float
            Tiempo de vida del decaimiento 2→1 (relajación no radiativa)
        tau10 : float
            Tiempo de vida del decaimiento 1→0 (emisión radiativa a 1530 nm)
        tau20 : float
            Tiempo de vida del decaimiento directo 2→0 (muy pequeño)
        """
        self.N0 = N0
        self.P_pump = P_pump
        self.tau21 = tau21  # Relajación no radiativa (nivel 2 → nivel 1)
        self.tau10 = tau10  # Emisión radiativa (nivel 1 → nivel 0) - 1530 nm
        self.tau20 = tau20  # Decaimiento directo (nivel 2 → nivel 0)
        
        # Longitudes de onda
        self.lambda_pump = 980e-9     # Bombeo (nm)
        self.lambda_emission = 1530e-9 # Emisión (nm)
        
        # Para el espectro de emisión
        self.emission_fwhm = 40e-9    # Ancho del espectro (FWHM) en metros
        
    def three_level_ode(self, y, t, pump_on=True):
        """
        Sistema de ecuaciones diferenciales para modelo de 3 niveles
        
        Niveles:
        0: Estado fundamental
        1: Estado metaestable (emisor a 1530 nm)
        2: Estado excitado superior (bombeo a 980 nm)
        """
        N0, N1, N2 = y
        
        # Tasa de bombeo (solo si está encendido)
        pump_rate = self.P_pump if pump_on else 0
        
        # Ecuaciones diferenciales acopladas
        dN0_dt = N1/self.tau10 + N2/self.tau20 - pump_rate * N0
        dN1_dt = N2/self.tau21 - N1/self.tau10
        dN2_dt = pump_rate * N0 - N2/self.tau21 - N2/self.tau20
        
        return [dN0_dt, dN1_dt, dN2_dt]
    
    def calculate_emission_rate(self, N1_array):
        """
        Calcula la tasa de emisión óptica desde el nivel 1 al nivel 0
        
        Parámetros:
        -----------
        N1_array : array
            Población del nivel 1 en función del tiempo
            
        Retorna:
        --------
        emission_rate : array
            Tasa de emisión en fotones/segundo
        """
        return N1_array / self.tau10
    
    def simulate_dynamics(self, t_total=0.1, dt=1e-5, laser_off_time=None):
        """
        Simula la dinámica del sistema de 3 niveles
        """
        t = np.arange(0, t_total, dt)
        
        # Condiciones iniciales: todos los iones en estado fundamental
        initial_conditions = [self.N0, 0, 0]
        
        if laser_off_time is None:
            # Simulación con láser encendido todo el tiempo
            solution = odeint(self.three_level_ode, initial_conditions, t, args=(True,))
            
        else:
            # Simulación con láser que se apaga
            t_on = t[t <= laser_off_time]
            t_off = t[t > laser_off_time]
            
            # Fase 1: Láser encendido
            sol_on = odeint(self.three_level_ode, initial_conditions, t_on, args=(True,))
            
            # Fase 2: Láser apagado
            initial_off = sol_on[-1]  # Estado al momento del apagado
            sol_off = odeint(self.three_level_ode, initial_off, 
                           t_off - laser_off_time, args=(False,))
            
            # Combinar soluciones
            solution = np.vstack([sol_on, sol_off])
        
        N0_array, N1_array, N2_array = solution.T
        
        return t, N0_array, N1_array, N2_array
    
    def generate_emission_spectrum(self, N1_population, wavelength_range=(1450e-9, 1610e-9), points=1000):
        """
        Genera el espectro de emisión como una gaussiana centrada en 1530 nm
        """
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], points)
        
        # Intensidad gaussiana centrada en 1530 nm
        sigma = self.emission_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Conversión FWHM → σ
        intensity = N1_population * np.exp(-0.5 * ((wavelengths - self.lambda_emission) / sigma)**2)
        
        return wavelengths * 1e9, intensity  # Convertir a nm para el gráfico