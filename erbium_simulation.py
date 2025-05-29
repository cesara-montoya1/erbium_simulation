import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy.constants import h, c
import matplotlib.patches as patches

# =============================================================================
# SIMULACIÓN AVANZADA DE IONES DE ERBIO (Er³⁺) - MODELO DE 3 NIVELES
# =============================================================================


class ErbiumSimulator:
    """
    Simulador avanzado para iones de Erbio con modelo de 3 niveles energéticos
    """

    def __init__(self, N0=1e24, P_pump=10e-3, tau21=1e-6, tau10=10e-3, tau20=None):
        """
        Inicializa el simulador con parámetros físicos

        N0      : densidad total de iones [iones/m³]
        P_pump  : potencia óptica del láser de bombeo [W]
        tau21   : tiempo de relajación no radiativa 2→1 [s]
        tau10   : vida del nivel metaestable 1→0 [s]
        tau20   : decaimiento directo 2→0 [s], opcional (default = 0.1 × tau21)
        """
        self.N0 = N0
        self.P_pump = P_pump
        self.tau21 = tau21
        self.tau10 = tau10
        self.tau20 = tau20 if tau20 is not None else 0.1 * tau21

        # Constantes físicas
        self.h = h
        self.c = c
        self.lambda_pump = 980e-9  # [m]
        self.lambda_emission = 1530e-9  # [m]
        self.sigma_a_p = 2.5e-24  # [m²] sección eficaz de absorción a 980 nm
        self.A_eff = 80e-12  # [m²] área efectiva del modo

        # Frecuencia del bombeo
        self.nu_pump = self.c / self.lambda_pump  # [Hz]

        # Tasa de bombeo
        self.W_pump = (self.sigma_a_p * self.P_pump) / (
            self.A_eff * self.h * self.nu_pump
        )  # [1/s]

        # Para el espectro de emisión
        self.emission_fwhm = 40e-9  # [m]

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

        Parámetros:
        -----------
        N1_population : float
            Número de iones en el nivel 1 (emisor)
        wavelength_range : tuple
            Rango de longitudes de onda en metros
        points : int
            Número de puntos de muestreo
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

    def _draw_energy_levels(self, ax, N0, N1, N2):
        """
        Dibuja diagrama de niveles energéticos con poblaciones
        """
        # Posiciones de los niveles
        levels = [1, 2, 3]
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
            "980 nm\n(bombeo)",
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
        ax.set_title(r"Diagrama de Niveles\n$Er^{3+}$")
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

        # TODO: cambiar por gráfica con dos escalas
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
        (line1_0,) = ax1.plot([], [], "b-", linewidth=2, label="Nivel 1")
        (line1_1,) = ax1.plot([], [], "r-", linewidth=2, label="Nivel 2")
        (line1_2,) = ax1.plot([], [], "g-", linewidth=2, label="Nivel 3")
        ax1.legend()

        (line3_1,) = ax3.plot([], [], "r-", linewidth=2, label="Nivel 2")
        (line3_2,) = ax3.plot([], [], "g-", linewidth=2, label="Nivel 3")
        ax3.legend()

        # Texto para mostrar tiempo actual
        time_text = ax1.text(
            0.02,
            0.98,
            "",
            transform=ax1.transAxes,
            verticalalignment="top",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Línea vertical para tiempo actual
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
            ax2.set_title("Niveles Energéticos")
            ax2.set_xticks([])
            ax2.grid(True, alpha=0.3)

            # Actualizar espectro
            ax4.clear()
            wavelengths, spectrum = self.generate_emission_spectrum(N1[frame])
            ax4.semilogy(wavelengths, spectrum, "r-", linewidth=2)
            ax4.fill_between(wavelengths, spectrum, alpha=0.3, color="red")
            ax4.set_xlim(1450, 1610)
            ax4.set_ylim(0, 24)
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

            # Actualizar texto
            time_text.set_text(
                f"Tiempo: {current_t * 1000:.1f} ms\nLáser: {laser_status}"
            )

            return [line1_0, line1_1, line1_2, line3_1, line3_2, time_text]

        # Crear animación (cada 10 frames para velocidad)
        frames = range(0, len(t), max(1, len(t) // 200))  # Limitar a 200 frames máximo
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
# EJECUCIÓN PRINCIPAL DE LA SIMULACIÓN AVANZADA
# =============================================================================


def main():
    print("=" * 70)
    print("SIMULACIÓN DE ABSORCIÓN Y EMISIÓN (Er³⁺) - MODELO DE 3 NIVELES")
    print("=" * 70)

    # Crear simulador con parámetros realistas
    simulator = ErbiumSimulator(
        N0=1e24,
        P_pump=10e-3,
        tau21=1e-6,
        tau10=10e-3,
    )

    print("Parámetros del modelo de 3 niveles:")
    print(f"• Número total de iones: {simulator.N0:.0e}")
    print(f"• Tiempo de vida 2→1 (relajación): {simulator.tau21 * 1e6:.1f} μs")
    print(f"• Tiempo de vida 1→0 (emisión): {simulator.tau10 * 1000:.1f} ms")
    print(f"• Tiempo de vida 2→0 (directo): {simulator.tau20 * 1e6:.1f} μs")
    print(f"• Potencia de bombeo: {simulator.P_pump}")

    # Simulación 1: Excitación continua
    print("\n1. Simulando modelo de 3 niveles - excitación continua...")
    t1, N0_1, N1_1, N2_1 = simulator.simulate_dynamics(t_total=0.05)

    # Simulación 2: Con apagado del láser
    print("2. Simulando con láser apagado a los 25 ms...")
    laser_off = 0.025
    t2, N0_2, N1_2, N2_2 = simulator.simulate_dynamics(
        t_total=0.05, laser_off_time=laser_off
    )

    print("3. Creando animación dinámica...")
    # Animación (usando datos con láser apagado para más interés visual)
    anim = simulator.create_animation(
        t2,
        N0_2,
        N1_2,
        N2_2,
        laser_off_time=laser_off,
        filename="erbium_3level_animation.gif",
    )

    # Análisis cuantitativo
    print("\n" + "=" * 70)
    print("ANÁLISIS DE RESULTADOS - MODELO DE 3 NIVELES")
    print("=" * 70)

    # Estado estacionario
    N1_steady = N1_1[-1]
    N2_steady = N2_1[-1]

    print("Estados estacionarios:")
    print(
        f"• Nivel 1 (emisor): {N1_steady:.0e} iones ({N1_steady / simulator.N0 * 100:.1f}%)"
    )
    print(
        f"• Nivel 2 (bombeo): {N2_steady:.0e} iones ({N2_steady / simulator.N0 * 100:.1f}%)"
    )

    # Tasa de emisión
    emission_rate = N1_steady / simulator.tau10
    print(f"• Tasa de emisión a 1530 nm: {emission_rate:.0e} fotones/(s × nm)")

    # Eficiencia cuántica
    pump_rate = simulator.P_pump * (simulator.N0 - N1_steady - N2_steady)
    quantum_efficiency = emission_rate / pump_rate if pump_rate > 0 else 0
    print(f"• Eficiencia cuántica: {quantum_efficiency * 100:.1f}%")

    print("\n" + "=" * 70)
    print("CARACTERÍSTICAS DEL ESPECTRO DE EMISIÓN")
    print("=" * 70)

    wavelengths, spectrum = simulator.generate_emission_spectrum(N1_steady)
    peak_idx = np.argmax(spectrum)
    fwhm_indices = np.where(spectrum >= spectrum[peak_idx] * 0.5)[0]
    fwhm = wavelengths[fwhm_indices[-1]] - wavelengths[fwhm_indices[0]]

    print(f"• Longitud de onda pico: {wavelengths[peak_idx]:.1f} nm")
    print(f"• FWHM del espectro: {fwhm:.1f} nm")
    print(f"• Intensidad pico: {spectrum[peak_idx]:.0e} u.a.")

    print("\n" + "=" * 70)
    print("VENTAJAS DEL MODELO DE 3 NIVELES")
    print("=" * 70)
    print("✓ Representa mejor la física real del Er³⁺")
    print("✓ Incluye relajación no radiativa (2→1)")
    print("✓ Explica la inversión de población efectiva")
    print("✓ Permite calcular eficiencias cuánticas realistas")
    print("✓ Predice espectros de emisión gaussianos")
    print("✓ Útil para diseño de amplificadores ópticos (EDFA)")

    plt.show()

    return simulator, anim


if __name__ == "__main__":
    simulator, animation = main()
    print("\n¡Simulación avanzada completada exitosamente!")
    print("Se han generado gráficos estáticos y animación dinámica.")
