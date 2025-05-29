"""
erbium_animation.py
===================
Funciones de animación para la simulación de iones de Erbio
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .erbium_plotting import draw_energy_levels

def create_animation(simulator, t, N0, N1, N2, filename='erbium_emission_animation.gif', 
                    laser_off_time=None, interval=50):
    """
    Crea animación de la dinámica temporal incluyendo tasa de emisión
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calcular tasa de emisión
    emission_rate = simulator.calculate_emission_rate(N1)
    
    # Configurar subplots
    ax1.set_xlim(0, t[-1] * 1000)
    ax1.set_ylim(0, simulator.N0 * 1.1)
    ax1.set_xlabel('Tiempo (ms)')
    ax1.set_ylabel('Población')
    ax1.set_title('Dinámica de Poblaciones')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlim(0, t[-1] * 1000)
    ax2.set_ylim(0, np.max(emission_rate) * 1.1)
    ax2.set_xlabel('Tiempo (ms)')
    ax2.set_ylabel('Tasa de emisión (fotones/s)')
    ax2.set_title('Tasa de Emisión Óptica')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.2, 2.2)
    ax3.set_ylabel('Energía (u.a.)')
    ax3.set_title('Niveles Energéticos')
    ax3.set_xticks([])
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlim(1450, 1610)
    ax4.set_xlabel('Longitud de onda (nm)')
    ax4.set_ylabel('Intensidad')
    ax4.set_title('Espectro de Emisión')
    ax4.grid(True, alpha=0.3)
    
    # Líneas para la animación
    line1_0, = ax1.plot([], [], 'b-', linewidth=2, label='Nivel 0')
    line1_1, = ax1.plot([], [], 'r-', linewidth=2, label='Nivel 1')
    line2_emission, = ax2.plot([], [], 'orange', linewidth=2, label='Tasa emisión')
    ax1.legend()
    ax2.legend()
    
    # Texto para mostrar tiempo actual y valores
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                       verticalalignment='top', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    emission_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, 
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
    
    # Línea vertical para tiempo actual
    if laser_off_time is not None:
        ax1.axvline(x=laser_off_time * 1000, color='gray', linestyle='--', 
                   linewidth=2, alpha=0.5, label='Láser apagado')
        ax2.axvline(x=laser_off_time * 1000, color='gray', linestyle='--', 
                   linewidth=2, alpha=0.5)
    
    def animate(frame):
        # Actualizar curvas temporales
        current_time = t[:frame+1] * 1000
        line1_0.set_data(current_time, N0[:frame+1])
        line1_1.set_data(current_time, N1[:frame+1])
        line2_emission.set_data(current_time, emission_rate[:frame+1])
        
        # Actualizar diagrama de niveles
        ax3.clear()
        draw_energy_levels(ax3, simulator, N0[frame], N1[frame], N2[frame])
        ax3.set_xlim(0, 1)
        ax3.set_ylim(-0.2, 2.2)
        ax3.set_ylabel('Energía (u.a.)')
        ax3.set_title('Niveles Energéticos')
        ax3.set_xticks([])
        ax3.grid(True, alpha=0.3)
        
        # Actualizar espectro
        ax4.clear()
        wavelengths, spectrum = simulator.generate_emission_spectrum(N1[frame])
        ax4.plot(wavelengths, spectrum, 'r-', linewidth=2)
        ax4.fill_between(wavelengths, spectrum, alpha=0.3, color='red')
        ax4.set_xlim(1450, 1610)
        ax4.set_ylim(0, np.max(spectrum) * 1.1 if np.max(spectrum) > 0 else 1e5)
        ax4.set_xlabel('Longitud de onda (nm)')
        ax4.set_ylabel('Intensidad')
        ax4.set_title('Espectro de Emisión')
        ax4.grid(True, alpha=0.3)
        
        # Determinar estado del láser
        current_t = t[frame]
        laser_status = "ON" if (laser_off_time is None or current_t <= laser_off_time) else "OFF"
        
        # Actualizar textos
        time_text.set_text(f'Tiempo: {current_t*1000:.1f} ms\nLáser: {laser_status}')
        emission_text.set_text(f'Emisión: {emission_rate[frame]:.1e} fotones/s\nN₁: {N1[frame]:.1e}')
        
        return [line1_0, line1_1, line2_emission, time_text, emission_text]
    
    # Crear animación
    frames = range(0, len(t), max(1, len(t)//200))  # Limitar a 200 frames máximo
    anim = FuncAnimation(fig, animate, frames=frames, interval=interval, 
                       blit=False, repeat=True)
    
    plt.tight_layout()
    
    # Guardar como GIF (opcional)
    try:
        anim.save(filename, writer='pillow', fps=20)
        print(f"Animación guardada como: {filename}")
    except:
        print("No se pudo guardar la animación (pillow no disponible)")
    
    return anim