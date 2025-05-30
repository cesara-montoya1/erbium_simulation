import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy.constants import h, c
import matplotlib.patches as patches
from scipy.interpolate import interp1d

# =============================================================================
# SIMULACIÓN AVANZADA DE IONES DE ERBIO (Er³⁺) - MODELO DE 3 NIVELES
# CON LONGITUD DE ONDA DE BOMBEO VARIABLE
# =============================================================================

max_intensity = 0

class ErbiumSimulator:
    """
    Simulador avanzado para iones de Erbio con modelo de 3 niveles energéticos
    y longitud de onda de bombeo variable
    """

    def __init__(self, N0=1e24, P_pump=10e-3, tau21=1e-6, tau10=10e-3, tau20=None, lambda_pump=980e-9):
        """
        Inicializa el simulador con parámetros físicos

        N0          : densidad total de iones [iones/m³]
        P_pump      : potencia óptica del láser de bombeo [W]
        tau21       : tiempo de relajación no radiativa 2→1 [s]
        tau10       : vida del nivel metaestable 1→0 [s]
        tau20       : decaimiento directo 2→0 [s], opcional (default = 0.1 × tau21)
        lambda_pump : longitud de onda de bombeo [m]
        """
        self.N0 = N0
        self.P_pump = P_pump
        self.tau21 = tau21
        self.tau10 = tau10
        self.tau20 = tau20 if tau20 is not None else 0.1 * tau21
        self.lambda_pump = lambda_pump

        # Constantes físicas
        self.h = h
        self.c = c
        self.lambda_emission = 1530e-9  # [m]
        self.A_eff = 80e-12  # [m²] área efectiva del modo

        # Datos experimentales de sección eficaz de absorción vs longitud de onda
        # Basados en datos reales del Er³⁺ en sílice
        self._setup_absorption_data()

        # Para el espectro de emisión
        self.emission_fwhm = 40e-9  # [m]

        # Actualizar parámetros dependientes de la longitud de onda
        self._update_wavelength_dependent_params()

    def _setup_absorption_data(self):
        """
        Configura los datos de sección eficaz de absorción del Er³⁺
        Datos basados en literatura científica
        """
        # Longitudes de onda en nm
        wavelengths_nm = np.array([
            400, 450, 500, 520, 540, 650, 800, 850, 900, 950, 975, 980, 985, 1000,
            1450, 1460, 1470, 1480, 1490, 1500, 1510, 1520, 1530, 1540, 1550, 1560
        ])
        
        # Sección eficaz de absorción en m² (datos aproximados del Er³⁺)
        sigma_abs_m2 = np.array([
            1e-26, 5e-26, 1e-25, 2e-25, 3e-25, 1e-25, 5e-25, 1e-24, 2e-24, 2.2e-24,
            2.5e-24, 2.8e-24, 2.5e-24, 2e-24, 1e-25, 2e-25, 4e-25, 6e-25, 8e-25,
            1e-24, 1.2e-24, 1.5e-24, 1.8e-24, 1.5e-24, 1.2e-24, 8e-25
        ])
        
        # Crear interpolador
        self.sigma_absorption_interp = interp1d(
            wavelengths_nm * 1e-9,  # Convertir a metros
            sigma_abs_m2,
            kind='cubic',
            bounds_error=False,
            fill_value=1e-26  # Valor mínimo para longitudes de onda fuera del rango
        )

    def get_absorption_cross_section(self, wavelength):
        """
        Obtiene la sección eficaz de absorción para una longitud de onda dada
        
        wavelength : float, longitud de onda en metros
        return     : float, sección eficaz en m²
        """
        return self.sigma_absorption_interp(wavelength)

    def set_pump_wavelength(self, lambda_pump):
        """
        Cambia la longitud de onda de bombeo y actualiza parámetros dependientes
        
        lambda_pump : float, nueva longitud de onda de bombeo en metros
        """
        self.lambda_pump = lambda_pump
        self._update_wavelength_dependent_params()

    def _update_wavelength_dependent_params(self):
        """
        Actualiza parámetros que dependen de la longitud de onda de bombeo
        """
        # Frecuencia del bombeo
        self.nu_pump = self.c / self.lambda_pump  # [Hz]
        
        # Sección eficaz de absorción para la longitud de onda actual
        self.sigma_a_p = self.get_absorption_cross_section(self.lambda_pump)  # [m²]
        
        # Tasa de bombeo
        self.W_pump = (self.sigma_a_p * self.P_pump) / (
            self.A_eff * self.h * self.nu_pump
        )  # [1/s]

    def three_level_ode(self, y, t, pump_on=True):
        N0, N1, N2 = y

        # Tasa de bombeo (1/s) solo si el láser está encendido
        pump_rate = self.W_pump if pump_on else 0

        # Ecuaciones de tasa acopladas
        dN0_dt = N1 / self.tau10 + N2 / self.tau20 - pump_rate * N0
        dN1_dt = N2 / self.tau21 - N1 / self.tau10
        dN2_dt = pump_rate * N0 - N2 / self.tau21 - N2 / self.tau20

        return [dN0_dt, dN1_dt, dN2_dt]

    def simulate_dynamics(self, t_total=0.1, dt=1e-5, laser_off_time=None):
        t = np.arange(0, t_total, dt)
        initial_conditions = [self.N0, 0, 0]

        if laser_off_time is None:
            solution = odeint(self.three_level_ode, initial_conditions, t, args=(True,))
        else:
            t_on = t[t <= laser_off_time]
            t_off = t[t > laser_off_time]
            sol_on = odeint(
                self.three_level_ode, initial_conditions, t_on, args=(True,)
            )
            sol_off = odeint(
                self.three_level_ode, sol_on[-1], t_off - laser_off_time, args=(False,)
            )
            solution = np.vstack([sol_on, sol_off])

        return t, solution[:, 0], solution[:, 1], solution[:, 2]

    def generate_emission_spectrum(
        self, N1_population, wavelength_range=(1450e-9, 1610e-9), points=1000
    ):
        """
        Genera el espectro de emisión física como número de fotones/(s × nm)
        """
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], points)

        # Tasa total de emisión [fotones/s]
        total_rate = N1_population / self.tau10

        # Gaussiana normalizada a área 1
        sigma = self.emission_fwhm / (2 * np.sqrt(2 * np.log(2)))
        G = np.exp(-0.5 * ((wavelengths - self.lambda_emission) / sigma) ** 2)
        G /= np.trapezoid(G, wavelengths)  # Normalización numérica sobre el dominio

        # Intensidad espectral [fotones/s/m]
        intensity_m = total_rate * G

        # Convertir a [fotones/s/nm]
        wavelengths_nm = wavelengths * 1e9
        intensity_nm = intensity_m * 1e-9

        return wavelengths_nm, intensity_nm

    def plot_absorption_spectrum(self):
        """
        Grafica el espectro de absorción del Er³⁺
        """
        wavelengths_nm = np.linspace(400, 1600, 1000)
        wavelengths_m = wavelengths_nm * 1e-9
        sigma_values = [self.get_absorption_cross_section(w) for w in wavelengths_m]
        
        plt.figure(figsize=(12, 6))
        plt.semilogy(wavelengths_nm, np.array(sigma_values) * 1e24, 'b-', linewidth=2)
        plt.axvline(self.lambda_pump * 1e9, color='red', linestyle='--', linewidth=2, 
                   label=f'Bombeo actual: {self.lambda_pump*1e9:.0f} nm')
        plt.xlabel('Longitud de onda [nm]')
        plt.ylabel('Sección eficaz de absorción [pm²]')
        plt.title('Espectro de Absorción del Er³⁺')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Marcar bandas importantes
        plt.axvspan(975, 985, alpha=0.2, color='green', label='Banda 980 nm')
        plt.axvspan(1450, 1600, alpha=0.2, color='orange', label='Banda 1530 nm')
        plt.axvspan(800, 850, alpha=0.2, color='purple', label='Banda 808 nm')
        plt.legend()
        
        return plt.gcf()

    def wavelength_sweep_analysis(self, wavelength_range=(800e-9, 1600e-9), points=50):
        """
        Analiza el comportamiento del sistema para diferentes longitudes de onda de bombeo
        """
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], points)
        steady_state_N1 = []
        steady_state_N2 = []
        emission_rates = []
        pump_efficiencies = []
        
        original_wavelength = self.lambda_pump  # Guardar la longitud de onda original
        
        for wavelength in wavelengths:
            self.set_pump_wavelength(wavelength)
            
            # Simular hasta estado estacionario
            t, N0, N1, N2 = self.simulate_dynamics(t_total=0.1)
            
            # Tomar valores del estado estacionario
            N1_steady = N1[-1]
            N2_steady = N2[-1]
            emission_rate = N1_steady / self.tau10
            
            # Calcular eficiencia de bombeo
            pump_power_absorbed = self.W_pump * N0[-1] * self.h * self.nu_pump
            pump_efficiency = (emission_rate * self.h * self.c / self.lambda_emission) / pump_power_absorbed if pump_power_absorbed > 0 else 0
            
            steady_state_N1.append(N1_steady)
            steady_state_N2.append(N2_steady)
            emission_rates.append(emission_rate)
            pump_efficiencies.append(pump_efficiency)
        
        # Restaurar longitud de onda original
        self.set_pump_wavelength(original_wavelength)
        
        return (wavelengths * 1e9, np.array(steady_state_N1), np.array(steady_state_N2), 
                np.array(emission_rates), np.array(pump_efficiencies))

    def plot_wavelength_sweep(self):
        """
        Crea gráficas del análisis de barrido de longitud de onda
        """
        wavelengths_nm, N1_steady, N2_steady, emission_rates, efficiencies = self.wavelength_sweep_analysis()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Población en estado estacionario
        ax1.plot(wavelengths_nm, N1_steady, 'r-', linewidth=2, label='Nivel 1 (emisor)')
        ax1.plot(wavelengths_nm, N2_steady, 'g-', linewidth=2, label='Nivel 2 (excitado)')
        ax1.axvline(self.lambda_pump * 1e9, color='black', linestyle='--', alpha=0.7, label='Bombeo actual')
        ax1.set_xlabel('Longitud de onda de bombeo [nm]')
        ax1.set_ylabel('Población en estado estacionario')
        ax1.set_title('Poblaciones vs Longitud de Onda de Bombeo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tasa de emisión
        ax2.semilogy(wavelengths_nm, emission_rates, 'r-', linewidth=2)
        ax2.axvline(self.lambda_pump * 1e9, color='black', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Longitud de onda de bombeo [nm]')
        ax2.set_ylabel('Tasa de emisión [fotones/s]')
        ax2.set_title('Intensidad de Emisión vs Longitud de Onda de Bombeo')
        ax2.grid(True, alpha=0.3)
        
        # Eficiencia de bombeo
        ax3.plot(wavelengths_nm, efficiencies * 100, 'b-', linewidth=2)
        ax3.axvline(self.lambda_pump * 1e9, color='black', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Longitud de onda de bombeo [nm]')
        ax3.set_ylabel('Eficiencia de bombeo [%]')
        ax3.set_title('Eficiencia vs Longitud de Onda de Bombeo')
        ax3.grid(True, alpha=0.3)
        
        # Sección eficaz de absorción
        wavelengths_abs = np.linspace(800, 1600, 500)
        sigma_values = [self.get_absorption_cross_section(w * 1e-9) for w in wavelengths_abs]
        ax4.semilogy(wavelengths_abs, np.array(sigma_values) * 1e24, 'purple', linewidth=2)
        ax4.axvline(self.lambda_pump * 1e9, color='black', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Longitud de onda [nm]')
        ax4.set_ylabel('Sección eficaz de absorción [pm²]')
        ax4.set_title('Espectro de Absorción del Er³⁺')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def compare_wavelengths(self, wavelengths_to_compare, t_total=0.1):
        """
        Compara la dinámica temporal para diferentes longitudes de onda
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        original_wavelength = self.lambda_pump
        
        for i, wavelength in enumerate(wavelengths_to_compare):
            self.set_pump_wavelength(wavelength * 1e-9)  # Convertir nm a m
            t, N0, N1, N2 = self.simulate_dynamics(t_total=t_total)
            
            color = colors[i % len(colors)]
            label = f'{wavelength} nm'
            
            # Poblaciones vs tiempo
            ax1.plot(t * 1000, N1, color=color, linewidth=2, label=f'N1 - {label}')
            ax2.plot(t * 1000, N2, color=color, linewidth=2, label=f'N2 - {label}')
            
            # Espectro de emisión final
            wavelengths_em, spectrum = self.generate_emission_spectrum(N1[-1])
            ax3.plot(wavelengths_em, spectrum, color=color, linewidth=2, label=label)
            
            # Tasa de bombeo vs tiempo
            pump_rate = self.W_pump * N0
            ax4.plot(t * 1000, pump_rate, color=color, linewidth=2, label=label)
        
        # Configurar subplots
        ax1.set_xlabel('Tiempo [ms]')
        ax1.set_ylabel('Población Nivel 1')
        ax1.set_title('Evolución Temporal - Nivel Emisor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Tiempo [ms]')
        ax2.set_ylabel('Población Nivel 2')
        ax2.set_title('Evolución Temporal - Nivel Excitado')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Longitud de onda [nm]')
        ax3.set_ylabel('Intensidad [u.a.]')
        ax3.set_title('Espectros de Emisión (Estado Estacionario)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_xlabel('Tiempo [ms]')
        ax4.set_ylabel('Tasa de bombeo [1/s]')
        ax4.set_title('Tasa de Bombeo vs Tiempo')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Restaurar longitud de onda original
        self.set_pump_wavelength(original_wavelength)
        
        return fig

    def _draw_energy_levels(self, ax, N0, N1, N2):
        """
        Dibuja diagrama de niveles energéticos con poblaciones
        """
        # Posiciones de los niveles
        levels = [0, 1, 2]
        energies = [0, 0.8, 1.3]  # Energías relativas

        # Dibujar niveles
        for i, (level, energy) in enumerate(zip(levels, energies)):
            populations = [N0, N1, N2]
            width = populations[i] / self.N0 * 0.4  # Ancho proporcional a la población

            # Línea del nivel energético
            ax.hlines(energy, 0.3, 0.7, colors="black", linewidth=2)

            # Rectángulo representando la población
            rect = patches.Rectangle(
                (0.5 - width / 2, energy - 0.05),
                width,
                0.1,
                linewidth=1,
                edgecolor="black",
                facecolor=["blue", "red", "green"][i],
                alpha=0.6,
            )
            ax.add_patch(rect)

            # Etiquetas
            ax.text(
                0.8,
                energy,
                f"Nivel {level}\nN={populations[i]:.1e}",
                va="center",
                fontsize=9,
            )

        # Flechas de transición
        # Bombeo (0 → 2)
        ax.annotate(
            "",
            xy=(0.15, 1.8),
            xytext=(0.15, 0),
            arrowprops=dict(arrowstyle="->", lw=2, color="green"),
        )
        ax.text(
            0.05,
            0.9,
            f"{self.lambda_pump*1e9:.0f} nm\n(bombeo)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
        )

        # Relajación (2 → 1)
        ax.annotate(
            "",
            xy=(0.25, 1),
            xytext=(0.25, 1.8),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="orange", linestyle="--"),
        )
        ax.text(
            0.27, 1.4, "relajación\nno radiativa", ha="left", va="center", fontsize=7
        )

        # Emisión (1 → 0)
        ax.annotate(
            "",
            xy=(0.35, 0),
            xytext=(0.35, 1),
            arrowprops=dict(arrowstyle="->", lw=2, color="red"),
        )
        ax.text(0.37, 0.5, "1530 nm\n(emisión)", ha="left", va="center", fontsize=8)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 2.2)
        ax.set_ylabel("Energía [eV]")
        ax.set_title(f"Diagrama de Niveles\n$Er^{{3+}}$ - Bombeo: {self.lambda_pump*1e9:.0f} nm")
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)

    def create_animation(
        self,
        t,
        N0,
        N1,
        N2,
        filename="erbium_animation.gif",
        laser_off_time=None,
        interval=50,
    ):
        """
        Crea animación de la dinámica temporal
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Configurar subplots
        ax1.set_xlim(0, t[-1] * 1000)
        ax1.set_ylim(0, self.N0 * 1.1)
        ax1.set_xlabel("Tiempo [ms]")
        ax1.set_ylabel("Población [iones/m³]")
        ax1.set_title("Dinámica de Poblaciones")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.2, 2)
        ax2.set_ylabel("Energía [eV]")
        ax2.set_title("Niveles Energéticos")
        ax2.set_xticks([])
        ax2.grid(True, alpha=0.3)

        ax3.set_xlim(0, t[-1] * 1000)
        ax3.set_ylim(1, self.N0)
        ax3.set_yscale("log")
        ax3.set_xlabel("Tiempo [ms]")
        ax3.set_ylabel("Población [iones/m³]")
        ax3.set_title("Estados Excitados")
        ax3.grid(True, alpha=0.3)

        ax4.set_xlim(1450, 1610)
        ax4.set_xlabel("Longitud de onda [nm]")
        ax4.set_ylabel("Intensidad [fotones/(s × nm)]")
        ax4.set_title("Espectro de Emisión")
        ax4.grid(True, alpha=0.3)

        # Líneas para la animación
        (line1_0,) = ax1.plot([], [], "b-", linewidth=2, label="Nivel 0")
        (line1_1,) = ax1.plot([], [], "r-", linewidth=2, label="Nivel 1")
        (line1_2,) = ax1.plot([], [], "g-", linewidth=2, label="Nivel 2")
        ax1.legend()

        (line3_1,) = ax3.plot([], [], "r-", linewidth=2, label="Nivel 1")
        (line3_2,) = ax3.plot([], [], "g-", linewidth=2, label="Nivel 2")
        ax3.legend()

        # Texto para mostrar información actual
        info_text = ax1.text(
            0.02,
            0.98,
            "",
            transform=ax1.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Línea vertical para tiempo de apagado del láser
        if laser_off_time is not None:
            ax1.axvline(
                x=laser_off_time * 1000,
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label="Láser apagado",
            )
            ax3.axvline(
                x=laser_off_time * 1000,
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
            )

        def animate(frame):
            global max_intensity
            
            # Actualizar curvas temporales
            current_time = t[: frame + 1] * 1000
            line1_0.set_data(current_time, N0[: frame + 1])
            line1_1.set_data(current_time, N1[: frame + 1])
            line1_2.set_data(current_time, N2[: frame + 1])
            line3_1.set_data(current_time, N1[: frame + 1])
            line3_2.set_data(current_time, N2[: frame + 1])

            # Actualizar diagrama de niveles
            ax2.clear()
            self._draw_energy_levels(ax2, N0[frame], N1[frame], N2[frame])
            ax2.set_xlim(0, 1)
            ax2.set_ylim(-0.2, 2.2)
            ax2.set_ylabel("Energía (eV)")
            ax2.set_title(f"Niveles Energéticos\nBombeo: {self.lambda_pump*1e9:.0f} nm")
            ax2.set_xticks([])
            ax2.grid(True, alpha=0.3)

            # Actualizar espectro
            ax4.clear()
            wavelengths, spectrum = self.generate_emission_spectrum(N1[frame])

            max_intensity = max(max_intensity, spectrum.max())

            ax4.plot(wavelengths, spectrum, "r-", linewidth=2)
            ax4.fill_between(wavelengths, spectrum, alpha=0.3, color="red")
            ax4.set_xlim(1450, 1610)
            ax4.set_ylim(0, max_intensity)
            ax4.set_xlabel("Longitud de onda [nm]")
            ax4.set_ylabel("Intensidad [fotones/(s × nm)]")
            ax4.set_title("Espectro de Emisión")
            ax4.grid(True, alpha=0.3)

            # Determinar estado del láser
            current_t = t[frame]
            laser_status = (
                "ON"
                if (laser_off_time is None or current_t <= laser_off_time)
                else "OFF"
            )

            # Actualizar texto informativo
            info_text.set_text(
                f"Tiempo: {current_t * 1000:.1f} ms\n"
                f"Láser: {laser_status}\n"
                f"Bombeo: {self.lambda_pump*1e9:.0f} nm\n"
                f"σ_abs: {self.sigma_a_p*1e24:.1f} pm²"
            )

            return [line1_0, line1_1, line1_2, line3_1, line3_2, info_text]

        # Crear animación
        frames = range(0, len(t), max(1, len(t) // 200))
        anim = FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=False, repeat=True
        )

        plt.tight_layout()

        # Guardar como GIF (opcional)
        try:
            anim.save(filename, writer="pillow", fps=30)
            print(f"Animación guardada como: {filename}")
        except Exception:
            print("No se pudo guardar la animación (pillow no disponible)")

        return anim


# =============================================================================
# EJECUCIÓN PRINCIPAL CON ANÁLISIS DE LONGITUD DE ONDA VARIABLE
# =============================================================================

def main():
    print("=" * 80)
    print("SIMULACIÓN DE Er³⁺ CON LONGITUD DE ONDA DE BOMBEO VARIABLE")
    print("=" * 80)

    # Crear simulador con parámetros realistas
    simulator = ErbiumSimulator(
        N0=1e24,
        P_pump=50e-3,
        tau21=1e-6,
        tau10=10e-3,
        lambda_pump=980e-9  # Longitud de onda inicial
    )

    print("Parámetros iniciales:")
    print(f"• Densidad total de iones: {simulator.N0:.0e} iones/m³")
    print(f"• Potencia de bombeo: {simulator.P_pump*1000:.1f} mW")
    print(f"• Longitud de onda de bombeo inicial: {simulator.lambda_pump*1e9:.0f} nm")
    print(f"• Sección eficaz a {simulator.lambda_pump*1e9:.0f} nm: {simulator.sigma_a_p*1e24:.2f} pm²")

    # 1. Mostrar espectro de absorción
    print("\n1. Generando espectro de absorción del Er³⁺...")
    fig_abs = simulator.plot_absorption_spectrum()
    
    # 2. Análisis de barrido de longitud de onda
    print("2. Realizando análisis de barrido de longitud de onda...")
    fig_sweep = simulator.plot_wavelength_sweep()
    
    # 3. Comparación entre longitudes de onda específicas
    print("3. Comparando longitudes de onda específicas...")
    wavelengths_to_compare = [808, 980, 1480]  # nm
    fig_compare = simulator.compare_wavelengths(wavelengths_to_compare)
    
    # 4. Simulación dinámica con longitud de onda específica
    print("4. Simulación dinámica con 980 nm...")
    simulator.set_pump_wavelength(980e-9)
    t, N0, N1, N2 = simulator.simulate_dynamics(t_total=0.05, laser_off_time=0.025)
    
    # 5. Crear animación
    print("5. Creando animación...")
    anim = simulator.create_animation(
        t, N0, N1, N2,
        laser_off_time=0.025,
        filename="erbium_variable_wavelength.gif"
    )
    
    # Análisis cuantitativo
    print("\n" + "=" * 80)
    print("ANÁLISIS CUANTITATIVO")
    print("=" * 80)
    
    # Comparar eficiencias para diferentes longitudes de onda
    test_wavelengths = [808, 980, 1480]  # nm
    print("Comparación de eficiencias:")
    print(f"{'Longitud de onda [nm]':<20} {'σ_abs [pm²]':<15} {'Población N1':<15} {'Tasa emisión [s⁻¹]':<20}")
    print("-" * 80)
    
    for wl in test_wavelengths:
        simulator.set_pump_wavelength(wl * 1e-9)
        t_test, N0_test, N1_test, N2_test = simulator.simulate_dynamics(t_total=0.1)
        N1_steady = N1_test[-1]
        emission_rate = N1_steady / simulator.tau10
        
        print(f"{wl:<20} {simulator.sigma_a_p*1e24:<15.2f} {N1_steady:<15.2e} {emission_rate:<20.2e}")
    
    # Encontrar longitud de onda óptima
    wavelengths_nm, N1_steady, N2_steady, emission_rates, efficiencies = simulator.wavelength_sweep_analysis()
    optimal_idx = np.argmax(emission_rates)
    optimal_wavelength = wavelengths_nm[optimal_idx]
    max_emission = emission_rates[optimal_idx]
    
    print(f"\nLongitud de onda óptima: {optimal_wavelength:.0f} nm")
    print(f"Máxima tasa de emisión: {max_emission:.2e} fotones/s")
    
    # Bandas de absorción importantes
    print("\n" + "=" * 80)
    print("BANDAS DE ABSORCIÓN IMPORTANTES DEL Er³⁺")
    print("=" * 80)
    print("• 808 nm: Banda de bombeo secundaria")
    print("• 980 nm: Banda de bombeo principal (más eficiente)")
    print("• 1480 nm: Banda de reabsorción/emisión")
    print("• 1530 nm: Banda de emisión principal")
    
    print("\nVentajas de usar 980 nm:")
    print("✓ Mayor sección eficaz de absorción")
    print("✓ Menor calentamiento del material")
    print("✓ Mejor inversión de población")
    print("✓ Estándar en la industria para EDFAs")
    
    print("\nDesventajas de otras longitudes de onda:")
    print("• 808 nm: Menor eficiencia, más calentamiento")
    print("• 1480 nm: Competencia con la emisión, menor ganancia neta")
    
    plt.show()
    
    return simulator, anim, fig_abs, fig_sweep, fig_compare

def interactive_wavelength_demo():
    """
    Función para demostración interactiva (opcional)
    """
    print("\n" + "=" * 80)
    print("DEMOSTRACIÓN INTERACTIVA")
    print("=" * 80)
    
    simulator = ErbiumSimulator(N0=1e24, P_pump=50e-3)
    
    test_wavelengths = [808, 850, 980, 1480, 1530]
    
    for wl in test_wavelengths:
        print(f"\nProbando longitud de onda: {wl} nm")
        simulator.set_pump_wavelength(wl * 1e-9)
        
        print(f"  Sección eficaz de absorción: {simulator.sigma_a_p*1e24:.2f} pm²")
        print(f"  Tasa de bombeo: {simulator.W_pump:.2e} s⁻¹")
        
        # Simulación rápida
        t, N0, N1, N2 = simulator.simulate_dynamics(t_total=0.05)
        N1_final = N1[-1]
        emission_rate = N1_final / simulator.tau10
        
        print(f"  Población final N1: {N1_final:.2e}")
        print(f"  Tasa de emisión: {emission_rate:.2e} fotones/s")
        print(f"  Eficiencia relativa: {emission_rate/1e20:.1f}%")

if __name__ == "__main__":
    # Ejecutar simulación principal
    simulator, animation, fig_abs, fig_sweep, fig_compare = main()
    
    # Ejecutar demostración interactiva
    interactive_wavelength_demo()
    
    print("\n" + "=" * 80)
    print("¡SIMULACIÓN COMPLETADA!")
    print("=" * 80)
    print("Se han generado las siguientes gráficas:")
    print("1. Espectro de absorción del Er³⁺")
    print("2. Análisis de barrido de longitud de onda")
    print("3. Comparación entre longitudes de onda específicas")
    print("4. Animación dinámica")
    print("\nFuncionalidades implementadas:")
    print("✓ Longitud de onda de bombeo variable")
    print("✓ Sección eficaz de absorción realista")
    print("✓ Análisis de eficiencia vs longitud de onda")
    print("✓ Comparación entre diferentes bombeos")
    print("✓ Identificación de longitud de onda óptima")
    print("✓ Datos basados en Er³⁺ real")