use dirs::home_dir;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Serialize, Deserialize, Clone)]
pub struct OrbitalElements {
    inclination: f64,
    ascending_node: f64,
    argument_of_periapsis: f64,
    semi_major_axis: f64,
    eccentricity: f64,
}

pub fn matmul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    c
}

pub fn get_possible_orbits() -> Vec<OrbitalElements> {
    let path = home_dir().unwrap().join("possible_orbits.json");
    let json_file = std::fs::File::open(path).unwrap();
    let deserialised: Vec<OrbitalElements> = serde_json::from_reader(json_file).unwrap();
    deserialised
}

impl OrbitalElements {
    pub fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        let cos_i = self.inclination.cos();
        let sin_i = self.inclination.sin();
        let cos_o = self.ascending_node.cos();
        let sin_o = self.ascending_node.sin();
        let cos_p = self.argument_of_periapsis.cos();
        let sin_p = self.argument_of_periapsis.sin();
        let epsilon = (23.44 as f64).to_radians();
        let cos_epsilon = epsilon.cos();
        let sin_epsilon = epsilon.sin();

        let inclination_matrix = [[1.0, 0.0, 0.0], [0.0, cos_i, -sin_i], [0.0, sin_i, cos_i]];

        let ascending_node_matrix = [[cos_o, -sin_o, 0.0], [sin_o, cos_o, 0.0], [0.0, 0.0, 1.0]];

        let argument_of_periapsis_matrix =
            [[cos_p, -sin_p, 0.0], [sin_p, cos_p, 0.0], [0.0, 0.0, 1.0]];

        let ecliptic_matrix = [
            [1.0, 0.0, 0.0],
            [0.0, cos_epsilon, sin_epsilon],
            [0.0, -sin_epsilon, cos_epsilon],
        ];

        let rotation_matrix = matmul(
            argument_of_periapsis_matrix,
            matmul(
                inclination_matrix,
                matmul(ascending_node_matrix, ecliptic_matrix),
            ),
        );
        rotation_matrix
    }
    pub fn get_query(&self) -> String {
        let rotation_matrix = self.to_rotation_matrix();
        let r00 = rotation_matrix[0][0];
        let r01 = rotation_matrix[0][1];
        let r02 = rotation_matrix[0][2];
        let r10 = rotation_matrix[1][0];
        let r11 = rotation_matrix[1][1];
        let r12 = rotation_matrix[1][2];
        let r20 = rotation_matrix[2][0];
        let r21 = rotation_matrix[2][1];
        let r22 = rotation_matrix[2][2];

        let conversion_query: String = format!(
            r#"
            WITH coords AS (
                SELECT
                    source_id, ra, dec, parallax,
                    (206265000/parallax) as d,
                    COS(dec * PI() / 180) * COS(ra * PI() / 180) * (206265000/parallax) as x,
                    COS(dec * PI() / 180) * SIN(ra * PI() / 180) * (206265000/parallax) as y,
                    SIN(dec * PI() / 180) * (206265000/parallax) as z
                FROM gaiadr3.gaia_source
            ),
            projected AS (
                SELECT 
                    coords.*,
                    {r00} * x + {r01} * y + {r02} * z as x_p,
                    {r10} * x + {r11} * y + {r12} * z as y_p,
                    {r20} * x + {r21} * y + {r22} * z as z_p
                FROM coords
            )
            SELECT TOP 30 *
            FROM projected
            WHERE ABS(z_p) < 10 
            "#
        );
        conversion_query
    }
}
