#!/usr/bin/env bash
# make_all.sh -- MOT lenh ve lai TOAN BO figure cua paper.
# (Y tuong tu plot_all_the_things.sh cua Git Re-Basin, ICLR'23.)
#
#   cd figures && bash make_all.sh
#
# Moi script duoi day CHI DOC CSV. Khong cai nao train hay do lai bat cu thu gi,
# nen chay bao nhieu lan cung an toan va mat vai giay. Moi script con in ra san
# mot \caption{...} de dan thang vao LaTeX -- so trong hinh va so trong caption
# vi the khong bao gio lech nhau.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Figure chinh ==="
python3 fig0_geometry_schematic.py    # Fig 1: hinh y niem -- trac dia vs day cung
python3 fig7_three_spaces.py          # (chua gan so): so do Cong thuc (4) -- ba khong gian, hai anh xa
python3 fig1_barrier_dF.py            # Fig 2: barrier + ||dF||_op, 3 regime x 3 kien truc
python3 fig2_exponent_scatter.py      # Fig 3: alpha_B vs 3 ung vien du bao
python3 fig3_rho_star.py              # Fig 4: rho* tai trung diem theo width
python3 fig4_rho_gate.py              # Fig 5: cong rho* -- khi nao xap xi bac hai dung
python3 fig5_path_profile.py          # Fig 6: do cong Fisher doc duong (kieu MEP)
python3 fig6_deviation.py             # Fig 7 (Muc 5.1): xi(t) do choi + dev_rel do that

echo
echo "=== Figure phu luc ==="
python3 figA2_by_activation.py        # A2: boc tach theo activation (chong lung Bang 1, Nhan xet 4.5)
python3 figA1_decompose.py            # A1: kiem chung Menh de 4.4 (Gauss-Newton vs van chuyen)

# --- viet tiep khi co du lieu ---
# python3 figA2_lambda_sweep.py       # do ben cua so mu theo damping lambda
# python3 figA3_full_scatter.py       # ban day du: cham het moi cap + phep do bi loai
# python3 figA4_numerics.py           # fd_instab, cg_resid

echo
echo "Xong. Hinh nam trong figures/out/  (.pdf cho paper, .png cho slide)"
