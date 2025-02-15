#include <iostream>

class Inflator {
private:
    bool invalve;      // State of the inlet valve: true (open), false (closed)
    bool outvalve;     // State of the outlet valve: true (open), false (closed)
    int psi;           // Current pressure in the system (PSI)
    int target_psi;    // Target pressure for the system (PSI)

public:
    // Constructor to initialize valve states, current pressure, and target pressure
    Inflator(bool initialInValve = false, bool initialOutValve = false, int initialPsi = 0, int initialTargetPsi = 0)
        : invalve(initialInValve), outvalve(initialOutValve), psi(initialPsi), target_psi(initialTargetPsi) {}

    // Getter for inlet valve state
    bool isInValveOpen() const {
        return invalve;
    }

    // Getter for outlet valve state
    bool isOutValveOpen() const {
        return outvalve;
    }

    // Getter for current pressure
    int getPsi() const {
        return psi;
    }

    // Getter for target pressure
    int getTargetPsi() const {
        return target_psi;
    }

    // Setters for controlling the inlet valve
    void openInValve() {
        if (!invalve) {
            invalve = true;
            std::cout << "Inlet valve is now open.\n";
        } else {
            std::cout << "Inlet valve is already open.\n";
        }
    }

    void closeInValve() {
        if (invalve) {
            invalve = false;
            std::cout << "Inlet valve is now closed.\n";
        } else {
            std::cout << "Inlet valve is already closed.\n";
        }
    }

    // Setters for controlling the outlet valve
    void openOutValve() {
        if (!outvalve) {
            outvalve = true;
            std::cout << "Outlet valve is now open.\n";
        } else {
            std::cout << "Outlet valve is already open.\n";
        }
    }

    void closeOutValve() {
        if (outvalve) {
            outvalve = false;
            std::cout << "Outlet valve is now closed.\n";
        } else {
            std::cout << "Outlet valve is already closed.\n";
        }
    }

    // Setter for current pressure
    void setPsi(int newPsi) {
        if (newPsi >= 0) {
            psi = newPsi;
            std::cout << "Current pressure set to " << psi << " PSI.\n";
        } else {
            std::cout << "Pressure cannot be negative.\n";
        }
    }

    // Setter for target pressure
    void setTargetPsi(int newTargetPsi) {
        if (newTargetPsi >= 0) {
            target_psi = newTargetPsi;
            std::cout << "Target pressure set to " << target_psi << " PSI.\n";
        } else {
            std::cout << "Target pressure cannot be negative.\n";
        }
    }
};

int main() {
    // Example usage of the Inflator class
    Inflator inflator; // Default constructor

    std::cout << "Initial inlet valve state: " << (inflator.isInValveOpen() ? "Open" : "Closed") << "\n";
    std::cout << "Initial outlet valve state: " << (inflator.isOutValveOpen() ? "Open" : "Closed") << "\n";
    std::cout << "Initial current pressure: " << inflator.getPsi() << " PSI\n";
    std::cout << "Initial target pressure: " << inflator.getTargetPsi() << " PSI\n";

    inflator.openInValve();
    inflator.setPsi(50); // Set current pressure
    inflator.setTargetPsi(100); // Set target pressure
    inflator.closeInValve();
    inflator.openOutValve();

    return 0;
}