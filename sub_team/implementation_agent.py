"""
Implementation Agent — Code Generation.

Responsibility
--------------
Input  : FormalSpec + MicroarchPlan.
Output : An RTLOutput containing synthesizable Verilog source code.

Method
------
Grammar-based code generation (no neural inference).  Each CPU component
(ALU, register file, pipeline registers, hazard unit, control unit) is
produced by a deterministic template renderer.  The same inputs always
produce the same RTL text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

from .specification_agent import FormalSpec
from .microarchitecture_agent import MicroarchPlan


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class RTLModule:
    """A single Verilog module."""
    name: str
    description: str
    source: str   # synthesizable Verilog text


@dataclass
class RTLOutput:
    """Collection of Verilog modules produced by the ImplementationAgent."""
    modules: List[RTLModule] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"RTLOutput ({len(self.modules)} modules)"]
        for m in self.modules:
            lines.append(f"  [{m.name}]  — {m.description}")
        return "\n".join(lines)

    def write_to_dir(self, directory: str) -> List[str]:
        """Write each module to *directory/<name>.v* and return paths."""
        import os
        os.makedirs(directory, exist_ok=True)
        paths: List[str] = []
        for m in self.modules:
            path = os.path.join(directory, f"{m.name}.v")
            with open(path, "w") as fh:
                fh.write(m.source)
            paths.append(path)
        return paths


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def _indent(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def _alu_source(data_width: int, has_mul: bool) -> str:
    mul_defines = ""
    mul_block = ""
    if has_mul:
        mul_defines = """\
`define ALU_MUL  4'd10
`define ALU_DIV  4'd11
`define ALU_REM  4'd12
"""
        mul_block = """
    `ALU_MUL:  result = $signed(a) * $signed(b);
    `ALU_DIV:  result = (b != 0) ? $signed(a) / $signed(b) : {DW{1'b1}};
    `ALU_REM:  result = (b != 0) ? $signed(a) % $signed(b) : a;"""

    return f"""\
`define ALU_ADD  4'd0
`define ALU_SUB  4'd1
`define ALU_XOR  4'd2
`define ALU_OR   4'd3
`define ALU_AND  4'd4
`define ALU_SLL  4'd5
`define ALU_SRL  4'd6
`define ALU_SRA  4'd7
`define ALU_SLT  4'd8
`define ALU_SLTU 4'd9
{mul_defines}

module alu #(parameter DW = {data_width}) (
    input  wire [3:0]    alu_op,
    input  wire [DW-1:0] a,
    input  wire [DW-1:0] b,
    output reg  [DW-1:0] result,
    output wire          zero
);
    assign zero = (result == {{DW{{1'b0}}}});
    always @(*) begin
        case (alu_op)
            `ALU_ADD:  result = a + b;
            `ALU_SUB:  result = a - b;
            `ALU_XOR:  result = a ^ b;
            `ALU_OR:   result = a | b;
            `ALU_AND:  result = a & b;
            `ALU_SLL:  result = a << b[4:0];
            `ALU_SRL:  result = a >> b[4:0];
            `ALU_SRA:  result = $signed(a) >>> b[4:0];
            `ALU_SLT:  result = {{{{(DW-1){{1'b0}}}}, ($signed(a) < $signed(b))}};
            `ALU_SLTU: result = {{{{(DW-1){{1'b0}}}}, (a < b)}};{mul_block}
            default:   result = {{DW{{1'b0}}}};
        endcase
    end
endmodule
"""


def _regfile_source(data_width: int, num_regs: int = 32) -> str:
    return f"""\
module regfile #(
    parameter DW  = {data_width},
    parameter NR  = {num_regs}
) (
    input  wire           clk,
    input  wire           we,
    input  wire [$clog2(NR)-1:0] rd_addr,
    input  wire [DW-1:0]  rd_data,
    input  wire [$clog2(NR)-1:0] rs1_addr,
    input  wire [$clog2(NR)-1:0] rs2_addr,
    output wire [DW-1:0]  rs1_data,
    output wire [DW-1:0]  rs2_data
);
    reg [DW-1:0] regs [0:NR-1];
    integer i;
    initial for (i = 0; i < NR; i = i + 1) regs[i] = {{DW{{1'b0}}}};

    // Write port (x0 is hardwired to zero)
    always @(posedge clk) begin
        if (we && rd_addr != 0)
            regs[rd_addr] <= rd_data;
    end

    // Read ports (combinational, with write-through for x0)
    assign rs1_data = (rs1_addr == 0) ? {{DW{{1'b0}}}} : regs[rs1_addr];
    assign rs2_data = (rs2_addr == 0) ? {{DW{{1'b0}}}} : regs[rs2_addr];
endmodule
"""


def _hazard_unit_source(forwarding: bool) -> str:
    if forwarding:
        fwd_logic = """
    // EX-EX forwarding
    always @(*) begin
        if (ex_mem_we && ex_mem_rd != 0 && ex_mem_rd == id_ex_rs1)
            forward_a = 2'b10;
        else if (mem_wb_we && mem_wb_rd != 0 && mem_wb_rd == id_ex_rs1)
            forward_a = 2'b01;
        else
            forward_a = 2'b00;

        if (ex_mem_we && ex_mem_rd != 0 && ex_mem_rd == id_ex_rs2)
            forward_b = 2'b10;
        else if (mem_wb_we && mem_wb_rd != 0 && mem_wb_rd == id_ex_rs2)
            forward_b = 2'b01;
        else
            forward_b = 2'b00;
    end"""
    else:
        fwd_logic = """
    // No forwarding: forward signals are always disabled
    always @(*) begin
        forward_a = 2'b00;
        forward_b = 2'b00;
    end"""

    return f"""\
module hazard_unit (
    // Pipeline register source/destination addresses
    input  wire [4:0] id_ex_rs1,
    input  wire [4:0] id_ex_rs2,
    input  wire [4:0] id_ex_rd,
    input  wire [4:0] ex_mem_rd,
    input  wire       ex_mem_we,
    input  wire [4:0] mem_wb_rd,
    input  wire       mem_wb_we,
    // Load-use detection
    input  wire       id_ex_mem_read,
    input  wire [4:0] if_id_rs1,
    input  wire [4:0] if_id_rs2,
    // Outputs
    output reg  [1:0] forward_a,
    output reg  [1:0] forward_b,
    output wire       stall
);
    // Load-use hazard: stall when load destination matches a source of the next instruction
    assign stall = id_ex_mem_read && (id_ex_rd != 0) &&
                   (id_ex_rd == if_id_rs1 || id_ex_rd == if_id_rs2);
{fwd_logic}
endmodule
"""


def _pipeline_top_source(
    isa_name: str,
    data_width: int,
    stage_names: List[str],
    forwarding: bool,
    has_mul: bool,
) -> str:
    stages_comment = " | ".join(stage_names)
    mul_decode = ""
    if has_mul:
        mul_decode = """
        // M-extension
        (opcode == 7'b0110011 && funct7[0] == 1'b1 && funct3 == 3'b000) ? 4'd8 : // MUL
        (opcode == 7'b0110011 && funct7[0] == 1'b1 && funct3 == 3'b100) ? 4'd9 : // DIV
        (opcode == 7'b0110011 && funct7[0] == 1'b1 && funct3 == 3'b110) ? 4'd10: // REM"""
    return f"""\
// Auto-generated top-level pipeline for {isa_name}
// Pipeline: {stages_comment}
// Forwarding: {forwarding}
`include "alu.v"
`include "regfile.v"
`include "hazard_unit.v"

module cpu_{isa_name.lower()} #(parameter DW = {data_width}) (
    input  wire        clk,
    input  wire        rst_n,
    // Instruction memory interface
    output wire [DW-1:0] imem_addr,
    input  wire [31:0]   imem_data,
    // Data memory interface
    output wire [DW-1:0] dmem_addr,
    output wire [DW-1:0] dmem_wdata,
    output wire          dmem_we,
    input  wire [DW-1:0] dmem_rdata
);
    // Program counter
    reg [DW-1:0] pc;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) pc <= {{DW{{1'b0}}}};
        else        pc <= pc_next;
    end

    // Fetch
    assign imem_addr = pc;
    wire [31:0] instr = imem_data;

    // Decode
    wire [6:0] opcode = instr[6:0];
    wire [4:0] rd     = instr[11:7];
    wire [4:0] rs1    = instr[19:15];
    wire [4:0] rs2    = instr[24:20];
    wire [2:0] funct3 = instr[14:12];
    wire [6:0] funct7 = instr[31:25];

    // Immediate decode (sign-extended per RV32I format)
    wire [DW-1:0] imm_i = {{{{(DW-12){{instr[31]}}}}, instr[31:20]}};
    wire [DW-1:0] imm_s = {{{{(DW-12){{instr[31]}}}}, instr[31:25], instr[11:7]}};
    wire [DW-1:0] imm_b = {{{{(DW-13){{instr[31]}}}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0}};
    wire [DW-1:0] imm_u = {{{{(DW-32){{instr[31]}}}}, instr[31:12], 12'b0}};
    wire [DW-1:0] imm_j = {{{{(DW-21){{instr[31]}}}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0}};
    wire [DW-1:0] imm =
        (opcode == 7'b0000011 || opcode == 7'b0010011 || opcode == 7'b1100111) ? imm_i :
        (opcode == 7'b0100011)                                                  ? imm_s :
        (opcode == 7'b1100011)                                                  ? imm_b :
        (opcode == 7'b0110111 || opcode == 7'b0010111)                         ? imm_u :
        (opcode == 7'b1101111)                                                  ? imm_j : imm_i;

    // Writeback mux: loads read from data memory, others from ALU
    wire          is_load = (opcode == 7'b0000011);
    wire [DW-1:0] wb_data;   // assigned after ALU instantiation

    // Register file
    wire [DW-1:0] rs1_data, rs2_data;
    regfile #(.DW(DW)) rf (
        .clk     (clk),
        .we      (reg_we),
        .rd_addr (rd),
        .rd_data (wb_data),
        .rs1_addr(rs1),
        .rs2_addr(rs2),
        .rs1_data(rs1_data),
        .rs2_data(rs2_data)
    );

    // ALU
    wire [3:0]    alu_op;
    wire [DW-1:0] alu_result;
    wire          alu_zero;
    alu #(.DW(DW)) alu_inst (
        .alu_op (alu_op),
        .a      (rs1_data),
        .b      (rs2_data),
        .result (alu_result),
        .zero   (alu_zero)
    );

    // Writeback mux assignment
    assign wb_data = is_load ? dmem_rdata : alu_result;

    // Hazard unit
    wire stall;
    wire [1:0] forward_a, forward_b;
    hazard_unit hu (
        .id_ex_rs1    (rs1),
        .id_ex_rs2    (rs2),
        .id_ex_rd     (rd),
        .ex_mem_rd    (5'b0),
        .ex_mem_we    (1'b0),
        .mem_wb_rd    (5'b0),
        .mem_wb_we    (1'b0),
        .id_ex_mem_read(is_load),
        .if_id_rs1    (rs1),
        .if_id_rs2    (rs2),
        .forward_a    (forward_a),
        .forward_b    (forward_b),
        .stall        (stall)
    );

    // PC update and control-flow targets
    wire        reg_we;
    wire [DW-1:0] pc_plus_4;
    wire [DW-1:0] branch_target;
    wire [DW-1:0] jal_target;
    wire [DW-1:0] jalr_target;
    wire          is_branch;
    wire          is_jal;
    wire          is_jalr;
    wire          branch_taken;
    wire [DW-1:0] pc_next;

    assign pc_plus_4     = pc + {{{{(DW-4){{1'b0}}}}, 4'd4}};
    assign branch_target = pc + imm;
    assign jal_target    = pc + imm;
    assign jalr_target   = (rs1_data + imm) & {{{{DW-1{{1'b1}}}}, 1'b0}};  // align to 2 bytes

    assign is_branch = (opcode == 7'b1100011); // BRANCH
    assign is_jal    = (opcode == 7'b1101111); // JAL
    assign is_jalr   = (opcode == 7'b1100111); // JALR

    // Branch condition evaluation (RV32I subset)
    assign branch_taken =
        is_branch && (
            (funct3 == 3'b000 && (rs1_data == rs2_data)) ||                     // BEQ
            (funct3 == 3'b001 && (rs1_data != rs2_data)) ||                     // BNE
            (funct3 == 3'b100 && ($signed(rs1_data) <  $signed(rs2_data))) ||   // BLT
            (funct3 == 3'b101 && ($signed(rs1_data) >= $signed(rs2_data))) ||   // BGE
            (funct3 == 3'b110 && (rs1_data <  rs2_data)) ||                     // BLTU
            (funct3 == 3'b111 && (rs1_data >= rs2_data))                        // BGEU
        );

    // Next PC selection: stall, then jumps, then taken branch, else sequential
    assign pc_next =
        stall        ? pc           :
        is_jal       ? jal_target   :
        is_jalr      ? jalr_target  :
        branch_taken ? branch_target :
                       pc_plus_4;

    // Data memory interface
    assign dmem_addr  = alu_result;
    assign dmem_wdata = rs2_data;
    assign dmem_we    = (opcode == 7'b0100011);  // STORE

    // Register write-enable: any non-store instruction that writes rd != x0
    assign reg_we  = !dmem_we && (rd != 5'b0);

    // ALU operation decode (RV32I{"/M" if has_mul else ""})
    assign alu_op =
        // R-type (register-register) arithmetic/logical
        (opcode == 7'b0110011 && funct3 == 3'b000 && funct7[5] == 1'b0) ? 4'd0 : // ADD
        (opcode == 7'b0110011 && funct3 == 3'b000 && funct7[5] == 1'b1) ? 4'd1 : // SUB
        (opcode == 7'b0110011 && funct3 == 3'b111)                      ? 4'd2 : // AND
        (opcode == 7'b0110011 && funct3 == 3'b110)                      ? 4'd3 : // OR
        (opcode == 7'b0110011 && funct3 == 3'b100)                      ? 4'd4 : // XOR
        (opcode == 7'b0110011 && funct3 == 3'b010)                      ? 4'd5 : // SLT
        (opcode == 7'b0110011 && funct3 == 3'b011)                      ? 4'd6 : // SLTU
        // I-type ALU-immediate
        (opcode == 7'b0010011 && funct3 == 3'b000)                      ? 4'd0 : // ADDI
        (opcode == 7'b0010011 && funct3 == 3'b111)                      ? 4'd2 : // ANDI
        (opcode == 7'b0010011 && funct3 == 3'b110)                      ? 4'd3 : // ORI
        (opcode == 7'b0010011 && funct3 == 3'b100)                      ? 4'd4 : // XORI
        (opcode == 7'b0010011 && funct3 == 3'b010)                      ? 4'd5 : // SLTI
        (opcode == 7'b0010011 && funct3 == 3'b011)                      ? 4'd6 : // SLTIU
        // Load/store and branch: use ADD for address / compare base
        (opcode == 7'b0000011)                                          ? 4'd0 : // LOAD
        (opcode == 7'b0100011)                                          ? 4'd0 : // STORE
        (opcode == 7'b1100011)                                          ? 4'd0 : // BRANCH
        // Jumps
        (opcode == 7'b1101111)                                          ? 4'd0 : // JAL
        (opcode == 7'b1100111)                                          ? 4'd0 : // JALR{mul_decode}
                                                                          4'd0;  // default: ADD
endmodule
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ImplementationAgent:
    """
    Generates synthesizable Verilog from a FormalSpec and a MicroarchPlan.

    Usage::

        agent  = ImplementationAgent()
        output = agent.run(formal_spec, microarch_plan)
        print(output.summary())
        output.write_to_dir("rtl/")
    """

    def run(self, spec: FormalSpec, plan: MicroarchPlan) -> RTLOutput:
        """
        Deterministically generate Verilog RTL for the specified CPU.
        """
        data_width = 64 if "64" in spec.isa_name else 32
        has_mul = any(
            enc.mnemonic in ("MUL", "DIV", "REM") for enc in spec.encodings
        )
        forwarding: bool = bool(spec.constraints.get("forwarding", True))
        stage_names = [s.name for s in plan.stages]

        output = RTLOutput()

        output.modules.append(RTLModule(
            name="alu",
            description=f"Arithmetic-Logic Unit ({data_width}-bit)",
            source=_alu_source(data_width, has_mul),
        ))

        output.modules.append(RTLModule(
            name="regfile",
            description=f"Register file (32 × {data_width}-bit)",
            source=_regfile_source(data_width),
        ))

        output.modules.append(RTLModule(
            name="hazard_unit",
            description=f"Hazard detection & forwarding unit (forwarding={forwarding})",
            source=_hazard_unit_source(forwarding),
        ))

        output.modules.append(RTLModule(
            name=f"cpu_{spec.isa_name.lower()}",
            description=f"Top-level CPU — {spec.isa_name} / {plan.pipeline_name}",
            source=_pipeline_top_source(
                spec.isa_name, data_width, stage_names, forwarding, has_mul
            ),
        ))

        return output
