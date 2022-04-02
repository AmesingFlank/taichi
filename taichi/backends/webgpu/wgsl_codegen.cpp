#include "taichi/backends/webgpu/wgsl_codegen.h"

#include <string>
#include <sstream>
#include <vector>

#include "taichi/program/program.h"
#include "taichi/program/kernel.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/ir.h"
#include "taichi/util/line_appender.h"
#include "taichi/backends/webgpu/kernel_utils.h"
#include "taichi/ir/transforms.h"
#include "taichi/math/arithmetic.h"

namespace taichi {
namespace lang {
namespace webgpu {

using ResourceInfo = TaskAttributes::ResourceInfo;
using ResourceType = TaskAttributes::ResourceType;
namespace {

class StringBuilder {
 public:
  const std::string &getString() {
    return buffer_;
  }

  template <typename T>
  StringBuilder &operator<<(const T &t) {
    std::ostringstream os;
    os << t;
    buffer_ += os.str();
    return *this;
  }

 private:
  std::string buffer_;
};

struct PointerInfo {
  bool is_root;
  int root_id;
};

void string_replace_all(std::string &str,
                        const std::string &from,
                        const std::string &to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();  // In case 'to' contains 'from', like replacing
                               // 'x' with 'yx'
  }
}

class TaskCodegen : public IRVisitor {
 public:
  struct Params {
    OffloadedStmt *task_ir;
    std::vector<CompiledSNodeStructs> compiled_structs;
    std::unordered_map<int, Texture *> textures;
    const KernelContextAttributes *ctx_attribs;
    std::string ti_kernel_name;
    int task_id_in_kernel;
    // vertex shader and fragment shaders share their buffer bindings
    // to avoid conflict, the fisrt binding used by frag shader should be
    // vert.bindings.size()
    int binding_point_begin;
  };

  const bool use_64bit_pointers = false;

  explicit TaskCodegen(const Params &params)
      : task_ir_(params.task_ir),
        compiled_structs_(params.compiled_structs),
        ctx_attribs_(params.ctx_attribs),
        task_name_(fmt::format("{}_t{:02d}",
                               params.ti_kernel_name,
                               params.task_id_in_kernel)),
        binding_point_begin_(params.binding_point_begin),
        textures_(params.textures) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    fill_snode_to_root();
  }

  void fill_snode_to_root() {
    for (int root = 0; root < compiled_structs_.size(); ++root) {
      for (auto [node_id, node] : compiled_structs_[root].snode_descriptors) {
        snode_to_root_[node_id] = root;
      }
    }
  }

  struct Result {
    std::string wgsl_code;
    TaskAttributes task_attribs;
  };

  Result run() {
    if (task_ir_->task_type == OffloadedTaskType::serial) {
      generate_serial_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::range_for) {
      // struct_for is automatically lowered to ranged_for for dense snodes
      generate_range_for_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::vertex_for) {
      generate_vertex_for_kernel(task_ir_);
    } else if (task_ir_->task_type == OffloadedTaskType::fragment_for) {
      generate_fragment_for_kernel(task_ir_);
    } else {
      TI_ERROR("Unsupported offload type={} on WGSL codegen",
               task_ir_->task_name());
    }
    Result res;
    res.wgsl_code = assemble_shader();
    res.task_attribs = std::move(task_attribs_);

    return res;
  }

  void visit(OffloadedStmt *) override {
    TI_ERROR("This codegen is supposed to deal with one offloaded task");
  }

  void visit(Block *stmt) override {
    for (auto &s : stmt->statements) {
      s->accept(this);
    }
  }

  void visit(ConstStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto &const_val = stmt->val[0];
    auto dt = const_val.dt.ptr_removed();
    emit_let(stmt->raw_name(), get_primitive_type_name(dt));
    if (dt->is_primitive(PrimitiveTypeID::f32)) {
      float f = const_val.val_float32();
      body_ << std::to_string(f) << "f";
    } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
      int i = const_val.val_int32();
      body_ << i;
    } else {
      TI_ERROR("unsupported const type");
    }

    body_ << ";\n";
  }

  void visit(RandStmt *stmt) override {
    init_rand();
    auto dt = stmt->element_type();
    auto dt_name = get_primitive_type_name(dt);
    emit_let(stmt->raw_name(), dt_name);
    if (dt->is_primitive(PrimitiveTypeID::i32)) {
      body_ << "rand_i32(gid3.x);\n";
    } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
      body_ << "rand_f32(gid3.x);\n";
    } else {
      TI_ERROR("unsupported prim type in rand")
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    const auto operand = stmt->operand->raw_name();

    const auto src_dt = stmt->operand->element_type();
    const auto dst_dt = stmt->element_type();
    std::string dst_dt_name = get_primitive_type_name(dst_dt);
    UnaryOpType op = stmt->op_type;

    std::string value;
    if (false) {
    }
#define HANDLE_FUNC_OP(op_name, func)                \
  else if (op == UnaryOpType::op_name) {             \
    value = std::string(func) + "(" + operand + ")"; \
  }
    HANDLE_FUNC_OP(sqrt, "sqrt")
    HANDLE_FUNC_OP(round, "round")
    HANDLE_FUNC_OP(floor, "floor")
    HANDLE_FUNC_OP(ceil, "ceil")
    HANDLE_FUNC_OP(abs, "abs")
    HANDLE_FUNC_OP(sgn, "sign")
    HANDLE_FUNC_OP(sin, "sin")
    HANDLE_FUNC_OP(asin, "asin")
    HANDLE_FUNC_OP(cos, "cos")
    HANDLE_FUNC_OP(acos, "acos")
    HANDLE_FUNC_OP(tan, "tan")
    HANDLE_FUNC_OP(tanh, "tanh")
    HANDLE_FUNC_OP(tan, "inv")
    HANDLE_FUNC_OP(exp, "exp")
    HANDLE_FUNC_OP(log, "log")
    HANDLE_FUNC_OP(rsqrt, "inverseSqrt")
#undef HANDLE_FUNC_OP
    else if (op == UnaryOpType::neg) {
      value = "-" + operand;
    }
    else if (op == UnaryOpType::logic_not) {
      std::string zero;
      if (src_dt->is_primitive(PrimitiveTypeID::f32)) {
        zero = "0.0f";
      } else if (src_dt->is_primitive(PrimitiveTypeID::i32)) {
        zero = "0";
      } else {
        TI_ERROR("unsupported prim type in unary op");
      }
      value = std::string(dst_dt_name) + "(" + operand + " == " + zero + ")";
    }
    else if (op == UnaryOpType::bit_not) {
      value = "~" + operand;
    }
    else if (op == UnaryOpType::inv || op == UnaryOpType::rcp) {
      value = "(1.0f / f32(" + operand + "))";
    }
    else if (op == UnaryOpType::cast_value) {
      value = std::string(dst_dt_name) + "(" + operand + ")";
    }
    else if (op == UnaryOpType::cast_bits) {
      value = "bitcast<" + std::string(dst_dt_name) + ">(" + operand + ")";
    }
    emit_let(stmt->raw_name(), dst_dt_name);
    value = std::string(dst_dt_name) + "(" + value + ")";
    body_ << value << ";\n";
  }

  void visit(BinaryOpStmt *stmt) override {
    const auto lhs = stmt->lhs->raw_name();
    const auto rhs = stmt->rhs->raw_name();
    const auto op = stmt->op_type;
    DataType dt = stmt->element_type();
    emit_let(stmt->raw_name(), get_primitive_type_name(dt));

    std::string value;
    if (false) {
    }
#define HANDLE_INFIX_OP(op_name, infix_token)    \
  else if (op == BinaryOpType::op_name) {        \
    value = lhs + " " + infix_token + " " + rhs; \
  }
#define HANDLE_INFIX_OP_U32(op_name, infix_token)          \
  else if (op == BinaryOpType::op_name) {                  \
    value = lhs + " " + infix_token + " u32(" + rhs + ")"; \
  }
#define HANDLE_FUNC_OP(op_name, func)                         \
  else if (op == BinaryOpType::op_name) {                     \
    value = std::string(func) + "(" + lhs + ", " + rhs + ")"; \
  }
    HANDLE_INFIX_OP(mul, "*")
    HANDLE_INFIX_OP(add, "+")
    HANDLE_INFIX_OP(sub, "-")
    HANDLE_INFIX_OP(mod, "%")
    HANDLE_INFIX_OP(bit_and, "&")
    HANDLE_INFIX_OP(bit_or, "|")
    HANDLE_INFIX_OP(bit_xor, "^")
    HANDLE_INFIX_OP_U32(bit_shl, "<<")
    HANDLE_INFIX_OP_U32(bit_shr, ">>")  // TODO: fix
    HANDLE_INFIX_OP_U32(bit_sar, ">>")  // TODO: fix
    HANDLE_INFIX_OP(cmp_lt, "<")
    HANDLE_INFIX_OP(cmp_le, "<=")
    HANDLE_INFIX_OP(cmp_gt, ">")
    HANDLE_INFIX_OP(cmp_ge, ">=")
    HANDLE_INFIX_OP(cmp_eq, "==")
    HANDLE_INFIX_OP(cmp_ne, "!=")
    HANDLE_FUNC_OP(pow, "pow")
    HANDLE_FUNC_OP(atan2, "atan2")
    HANDLE_FUNC_OP(max, "max")
    HANDLE_FUNC_OP(min, "min")
#undef HANDLE_INFIX_OP
#undef HANDLE_FUNC_OP
    else if (op == BinaryOpType ::div) {
      value = lhs + " / " + rhs;
    }
    else if (op == BinaryOpType ::truediv) {
      value = "1.0 * " + lhs + " / " + rhs;
    }
    else if (op == BinaryOpType ::floordiv) {
      value = "floor(1.0 * " + lhs + " / " + rhs + ")";
    }

    value = std::string(get_primitive_type_name(dt)) + "(" + value + ")";

    body_ << value << ";\n";
  }

  void visit(RangeForStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit_var(stmt->raw_name(), "i32");
    body_ << stmt->begin->raw_name() << ";\n";
    body_ << body_indent() << "loop {\n";
    indent();
    body_ << body_indent() << "if (" << stmt->raw_name()
          << " >= " << stmt->end->raw_name() << "){ break; }\n";

    stmt->body->accept(this);

    body_ << body_indent() << stmt->raw_name() << " = " << stmt->raw_name()
          << " + 1;\n";
    dedent();
    body_ << body_indent() << "}\n";
  }

  void visit(IfStmt *stmt) override {
    body_ << body_indent() << "if (bool(" << stmt->cond->raw_name() << ")){\n";
    indent();

    if (stmt->true_statements) {
      stmt->true_statements->accept(this);
    }

    dedent();
    body_ << body_indent() << "}\n";
    if (stmt->false_statements) {
      body_ << body_indent() << "else {\n";
      indent();
      stmt->false_statements->accept(this);
      dedent();
      body_ << body_indent() << "}\n";
    }
  }

  void visit(WhileControlStmt *stmt) override {
    body_ << body_indent() << "break;\n";
  }

  void visit(ContinueStmt *stmt) override {
    auto stmt_in_off_for = [stmt]() {
      TI_ASSERT(stmt->scope != nullptr);
      if (auto *offl = stmt->scope->cast<OffloadedStmt>(); offl) {
        TI_ASSERT(offl->task_type == OffloadedStmt::TaskType::range_for ||
                  offl->task_type == OffloadedStmt::TaskType::struct_for);
        return true;
      }
      return false;
    };
    if (stmt_in_off_for()) {
      // then this parallel task is done;
      body_ << body_indent() << "ii = ii + total_invocs;\n";
      // continue in grid-strided loop;
      body_ << body_indent() << "continue;\n";
    } else {
      body_ << body_indent() << "continue;\n";
    }
  }

  void visit(WhileStmt *stmt) override {
    body_ << body_indent() << "loop {\n";
    indent();
    stmt->body->accept(this);
    dedent();
    body_ << body_indent() << "}\n";
  }

  void visit(VertexInputStmt *stmt) override {
    int loc = stmt->location;
    std::string dt_name = get_primitive_type_name(stmt->element_type());
    std::string input_name =
        std::string("in_") + std::to_string(loc) + "_" + dt_name;
    ensure_stage_in_struct();
    add_stage_in_member(input_name, dt_name, loc);
    emit_let(stmt->raw_name(), dt_name);
    body_ << body_indent() << "stage_input." << input_name << ";\n";
  }

  void visit(FragmentInputStmt *stmt) override {
    int loc = stmt->location;
    std::string dt_name = get_primitive_type_name(stmt->element_type());
    std::string input_name =
        std::string("in_") + std::to_string(loc) + "_" + dt_name;
    ensure_stage_in_struct();
    add_stage_in_member(input_name, dt_name, loc);
    emit_let(stmt->raw_name(), dt_name);
    body_ << body_indent() << "stage_input." << input_name << ";\n";
  }

  void visit(VertexOutputStmt *stmt) override {
    int loc = stmt->location;
    std::string dt_name = get_primitive_type_name(stmt->value->element_type());
    std::string output_name =
        std::string("out_") + std::to_string(loc) + "_" + dt_name;
    ensure_stage_out_struct();
    add_stage_out_member(output_name, dt_name, loc);

    body_ << body_indent() << "stage_output." << output_name << "="
          << stmt->value->raw_name() << ";\n";
  }

  void visit(BuiltInOutputStmt *stmt) override {
    ensure_stage_out_struct();
    int num_components = stmt->values.size();
    DataType prim_type = stmt->values[0]->element_type();
    std::string type_name =
        get_scalar_or_vector_type_name(prim_type, num_components);

    std::string output_expr =
        get_scalar_or_vector_expr(stmt->values, type_name);

    std::string output_name;

    if (stmt->built_in == BuiltInOutputStmt::BuiltIn::Color) {
      int loc = stmt->location;
      output_name = std::string("color_") + std::to_string(loc);
      add_stage_out_member(output_name, type_name, loc);
    } else if (stmt->built_in == BuiltInOutputStmt::BuiltIn::Position) {
      int loc = stmt->location;
      output_name = std::string("position");
      add_stage_out_builtin_member(output_name, type_name, "position");
    } else if (stmt->built_in == BuiltInOutputStmt::BuiltIn::FragDepth) {
      int loc = stmt->location;
      output_name = std::string("frag_depth");
      add_stage_out_builtin_member(output_name, type_name, "frag_depth");
    }

    body_ << body_indent() << "stage_output." << output_name << "="
          << output_expr << ";\n";
  }

  void visit(DiscardStmt *stmt) override {
    body_ << body_indent() << "discard;\n";
  }

  void visit(TextureFunctionStmt *stmt) override {
    Texture *texture = stmt->texture;
    ResourceInfo texture_resource = {ResourceType::Texture, texture->id};
    bool requires_sampler;
    std::string func_name;
    switch (stmt->func) {
      case TextureFunctionStmt::Function::Sample: {
        requires_sampler = true;
        func_name = "textureSample";
        break;
      }
      case TextureFunctionStmt::Function::Load: {
        requires_sampler = false;
        func_name = "textureLoad";
        break;
      }
      case TextureFunctionStmt::Function::Store: {
        requires_sampler = false;
        texture_resource.type = ResourceType::StorageTexture;
        func_name = "textureStore";
        break;
      }
      default: {
        TI_ERROR("unsupported texture fun")
      }
    }
    std::string texture_name = get_texture_name(texture_resource);
    std::string texel_type_name =
        get_scalar_or_vector_type_name(texture->params.primitive, 4);

    ResourceInfo sampler_resource = {ResourceType::Sampler, texture->id};
    std::string sampler_name;
    if (requires_sampler) {
      sampler_name = get_sampler_name(sampler_resource);
    }
    DataType coords_prim_type = stmt->operand_values[0]->element_type();
    int coords_component_count =
        get_texture_coords_num_components(texture->params.dimensionality);
    std::string coords_type_name = get_scalar_or_vector_type_name(
        coords_prim_type, coords_component_count);
    auto coords_stmts = std::vector<Stmt *>(
        stmt->operand_values.begin(),
        stmt->operand_values.begin() + coords_component_count);
    std::string coords_expr =
        get_scalar_or_vector_expr(coords_stmts, coords_type_name);
    auto non_coords_stmts = std::vector<Stmt *>(
        stmt->operand_values.begin() + coords_component_count,
        stmt->operand_values.end());
    switch (stmt->func) {
      case TextureFunctionStmt::Function::Sample: {
        emit_let(stmt->raw_name(), texel_type_name);
        body_ << "textureSample(" << texture_name << ", " << sampler_name
              << ", " << coords_expr << ");\n";
        break;
      }
      case TextureFunctionStmt::Function::Load: {
        emit_let(stmt->raw_name(), texel_type_name);
        body_ << "textureLoad(" << texture_name << ", " << coords_expr
              << ", 0);\n";
        break;
      }
      case TextureFunctionStmt::Function::Store: {
        auto &value_stmts = non_coords_stmts;
        DataType value_prim_type = value_stmts[0]->element_type();
        std::string value_type_name =
            get_scalar_or_vector_type_name(value_prim_type, value_stmts.size());
        std::string value_expr =
            get_scalar_or_vector_expr(value_stmts, value_type_name);
        body_ << body_indent() << "textureStore(" << texture_name << ", "
              << coords_expr << ", " << value_expr << ");\n";
        break;
      }
      default: {
        TI_ERROR("unsupported texture fun");
      }
    }
  }

  void visit(CompositeExtractStmt *stmt) override {
    std::string type_name = get_primitive_type_name(stmt->element_type());
    emit_let(stmt->raw_name(), type_name);
    body_ << stmt->base->raw_name();
    switch (stmt->element_index) {
      case 0: {
        body_ << ".x";
        break;
      }
      case 1: {
        body_ << ".y";
        break;
      }
      case 2: {
        body_ << ".z";
        break;
      }
      case 3: {
        body_ << ".w";
        break;
      }
      default: {
        TI_ERROR("unsupported composite extract index: {}",
                 stmt->element_index);
      }
    }
    body_ << ";\n";
  }

  void visit(ArgLoadStmt *stmt) override {
    const auto arg_id = stmt->arg_id;
    const auto &arg_attribs = ctx_attribs_->args()[arg_id];
    const auto offset_in_mem = arg_attribs.offset_in_mem;
    if (stmt->is_ptr) {
      TI_ERROR("arg is ptr... what does that mean lol");
    } else {
      const auto dt = arg_attribs.dt;
      std::string buffer_name =
          get_buffer_member_name(ResourceInfo(ResourceType::Args));
      if (!enforce_16bytes_alignment()) {
        emit_let(stmt->raw_name(), get_primitive_type_name(dt));
        body_ << "bitcast<" << get_primitive_type_name(dt) << ">("
              << buffer_name << "["
              << std::to_string(offset_in_mem / get_raw_data_type_size())
              << "]);\n";
      } else {
        auto temp = get_temp();
        emit_let(temp, "i32");
        body_ << "find_vec4_component(" << buffer_name << "["
              << std::to_string(offset_in_mem) << get_raw_data_index_shift()
              << "], " << std::to_string(offset_in_mem) << ");\n";
        emit_let(stmt->raw_name(), get_primitive_type_name(dt));
        body_ << "bitcast<" << get_primitive_type_name(dt) << ">(" << temp
              << ");\n";
      }
    }
  }

  void visit(ReturnStmt *stmt) override {
    if (enforce_16bytes_alignment()) {
      TI_ERROR("Ret cannot be used while enforcing 16 bytes alignment")
    }
    for (int i = 0; i < stmt->values.size(); i++) {
      body_ << body_indent()
            << get_buffer_member_name(ResourceInfo(ResourceType::Rets)) << "["
            << std::to_string(i) << "] = ";
      auto dt = stmt->element_types()[i];
      if (dt->is_primitive(PrimitiveTypeID::f32)) {
        body_ << "bitcast<i32>(" << stmt->values[i]->raw_name() << ");\n";
      } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
        body_ << stmt->values[i]->raw_name() << ";\n";
      } else {
        TI_ERROR("unsupported primitive type in return");
      }
    }
  }

  void visit(AllocaStmt *stmt) override {
    auto dt = stmt->element_type();
    // not using emit_var() because it emits an extra equals token..
    body_ << body_indent() << "var " << stmt->raw_name() << " : "
          << get_primitive_type_name(dt) << ";\n";
  }

  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(stmt->src.size() == 1);
    TI_ASSERT(stmt->src[0].offset == 0);
    auto dt = stmt->element_type();
    emit_let(stmt->raw_name(), get_primitive_type_name(dt));
    body_ << stmt->src[0].var->raw_name() << ";\n";
  }

  void visit(LocalStoreStmt *stmt) override {
    body_ << body_indent() << stmt->dest->raw_name() << " = "
          << stmt->val->raw_name() << ";\n";
  }

  void visit(GetRootStmt *stmt) override {
    const int root_id = snode_to_root_.at(stmt->root()->id);
    emit_let(stmt->raw_name(), get_pointer_int_type_name());
    body_ << "0;\n";
    pointer_infos_[stmt->raw_name()] = {true, root_id};
  }

  void visit(GetChStmt *stmt) override {
    // TODO: GetChStmt -> GetComponentStmt ?
    const int root = snode_to_root_.at(stmt->input_snode->id);

    const auto &snode_descs = compiled_structs_[root].snode_descriptors;
    auto *out_snode = stmt->output_snode;
    TI_ASSERT(snode_descs.at(stmt->input_snode->id).get_child(stmt->chid) ==
              out_snode);

    const auto &desc = snode_descs.at(out_snode->id);
    emit_let(stmt->raw_name(), get_pointer_int_type_name());
    body_ << stmt->input_ptr->raw_name() << " + "
          << (desc.mem_offset_in_parent_cell / 4) << ";\n";
    pointer_infos_[stmt->raw_name()] = {true, root};
  }

  void visit(SNodeLookupStmt *stmt) override {
    // TODO: SNodeLookupStmt -> GetSNodeCellStmt ?
    bool is_root{false};  // Eliminate first root snode access
    const int root_id = snode_to_root_.at(stmt->snode->id);
    std::string parent;

    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      TI_ASSERT(root_stmts_.at(root_id) != nullptr);
      parent = root_stmts_.at(root_id)->raw_name();
    }
    const auto *sn = stmt->snode;

    const auto &snode_descs = compiled_structs_[root_id].snode_descriptors;
    const auto &desc = snode_descs.at(sn->id);
    emit_let(stmt->raw_name(), get_pointer_int_type_name());
    body_ << parent << " + (" << (desc.cell_stride / 4) << " * "
          << (stmt->input_index->raw_name()) << ");\n";
    pointer_infos_[stmt->raw_name()] = {true, root_id};
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    emit_let(stmt->raw_name(), get_pointer_int_type_name());
    body_ << (stmt->offset) / 4 << ";\n";
    pointer_infos_[stmt->raw_name()] = {false, -1};
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (enforce_16bytes_alignment()) {
      TI_ERROR("global store cannot be used while enforcing 16 bytes alignment")
    }
    TI_ASSERT(stmt->width() == 1);
    PointerInfo info = pointer_infos_.at(stmt->dest->raw_name());
    std::string buffer_name;
    if (info.is_root) {
      int root_id = info.root_id;
      buffer_name = get_buffer_member_name(
          ResourceInfo(ResourceType::RootNormal, root_id));
    } else {
      buffer_name =
          get_buffer_member_name(ResourceInfo(ResourceType::GlobalTemps));
    }
    body_ << body_indent() << buffer_name << "[" << stmt->dest->raw_name()
          << "] = bitcast<i32>(" << (stmt->val->raw_name()) << ");\n";
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto dt = stmt->element_type();
    PointerInfo info = pointer_infos_.at(stmt->src->raw_name());
    std::string buffer_name;
    if (info.is_root) {
      int root_id = info.root_id;
      buffer_name = get_buffer_member_name(
          ResourceInfo(ResourceType::RootNormal, root_id));
    } else {
      buffer_name =
          get_buffer_member_name(ResourceInfo(ResourceType::GlobalTemps));
    }
    if (!enforce_16bytes_alignment()) {
      emit_let(stmt->raw_name(), get_primitive_type_name(dt));
      body_ << "bitcast<" << get_primitive_type_name(dt) << ">(" << buffer_name
            << "[" << stmt->src->raw_name() << "]);\n";
    } else {
      auto temp = get_temp();
      emit_let(temp, "i32");
      body_ << "find_vec4_component(" << buffer_name << "["
            << stmt->src->raw_name() << get_raw_data_index_shift() << "], "
            << stmt->src->raw_name() << ");\n";
      emit_let(stmt->raw_name(), get_primitive_type_name(dt));
      body_ << "bitcast<" << get_primitive_type_name(dt) << ">(" << temp
            << ");\n";
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    const auto dt = stmt->dest->element_type().ptr_removed();
    std::string dt_name = get_primitive_type_name(dt);
    std::string buffer_member_name;
    PointerInfo info = pointer_infos_.at(stmt->dest->raw_name());
    TI_ASSERT(info.is_root);
    int root_id = info.root_id;
    buffer_member_name = get_buffer_member_name(
        ResourceInfo(ResourceType::RootAtomicI32, root_id));
    std::string atomic_func_name;
    switch (stmt->op_type) {
      case AtomicOpType::add: {
        atomic_func_name = "atomicAdd";
        break;
      }
      case AtomicOpType::sub: {
        atomic_func_name = "atomicSub";
        break;
      }
      case AtomicOpType::max: {
        atomic_func_name = "atomicMax";
        break;
      }
      case AtomicOpType::min: {
        atomic_func_name = "atomicMin";
        break;
      }
      case AtomicOpType::bit_and: {
        atomic_func_name = "atomicAnd";
        break;
      }
      case AtomicOpType::bit_or: {
        atomic_func_name = "atomicOr";
        break;
      }
      case AtomicOpType::bit_xor: {
        atomic_func_name = "atomicXor";
        break;
      }
    }

    /*
fn atomicAddFloat(dest: ptr<storage, atomic<i32>, read_write>, v: f32) -> f32 {
  loop {
    let old_val : f32 = bitcast<f32>(atomicLoad(dest));
    let new_val : f32 = old_val + v;
    if(atomicCompareExchangeWeak(dest, bitcast<i32>(old_val),
bitcast<i32>(new_val)).y != 0){ return old_val;
    }
  }
}
*/

    // WGSL doesn't allow declaring a function whose argument is a pointer to
    // SSBO... so we inline it

    auto result = get_temp("atomic_op_result");

    body_ << body_indent() << "var " << result << " : " << dt_name << ";\n";

    std::string ptr =
        "&(" + buffer_member_name + "[" + stmt->dest->raw_name() + "])";

    if (dt->is_primitive(PrimitiveTypeID::i32)) {
      body_ << body_indent() << result << " = " << atomic_func_name << "("
            << ptr << ", " << stmt->val->raw_name() << ");\n";
    } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
      body_ << body_indent() << "loop {\n";
      indent();
      std::string old_val = get_temp("old_val");
      emit_let(old_val, "f32");
      body_ << "bitcast<f32>(atomicLoad(" << ptr << "));\n";

      std::string new_val_expr;
      switch (stmt->op_type) {
        case AtomicOpType::add: {
          new_val_expr = old_val + " + " + stmt->val->raw_name();
          break;
        }
        case AtomicOpType::sub: {
          new_val_expr = old_val + " - " + stmt->val->raw_name();
          break;
        }
        case AtomicOpType::max: {
          new_val_expr = "max(" + old_val + ", " + stmt->val->raw_name() + ")";
          break;
        }
        case AtomicOpType::min: {
          new_val_expr = "min(" + old_val + ", " + stmt->val->raw_name() + ")";
          break;
        }
        default:
          TI_ERROR("unsupported atomic op for f32");
      }

      std::string new_val = get_temp("new_val");
      emit_let(new_val, "f32");
      body_ << new_val_expr << ";\n";
      body_ << body_indent() << "if(atomicCompareExchangeWeak(" << ptr
            << ", bitcast<i32>(" << old_val << "), bitcast<i32>(" << new_val
            << ")).y!=0){\n";
      indent();
      body_ << body_indent() << result << " = " << old_val << ";\n";
      body_ << body_indent() << "break;\n";
      dedent();
      body_ << body_indent() << "}\n";
      dedent();
      body_ << body_indent() << "}\n";
    } else {
      TI_ERROR("unsupported prim type in atomic op")
    }
    // body_<<body_indent()<<"storageBarrier();\n";
    emit_let(stmt->raw_name(), dt_name);
    body_ << result << ";\n";
  }

  void visit(LoopIndexStmt *stmt) override {
    const auto stmt_name = stmt->raw_name();
    if (stmt->loop->is<OffloadedStmt>()) {
      const auto type = stmt->loop->as<OffloadedStmt>()->task_type;
      if (type == OffloadedTaskType::range_for) {
        TI_ASSERT(stmt->index == 0);
        emit_let(stmt->raw_name(), "i32");
        body_ << "ii;\n";
      } else {
        TI_NOT_IMPLEMENTED;
      }
    } else if (stmt->loop->is<RangeForStmt>()) {
      TI_ASSERT(stmt->index == 0);
      emit_let(stmt->raw_name(), "i32");
      body_ << stmt->loop->raw_name() << ";\n";
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

 private:
  void emit_let(std::string name, std::string type) {
    body_ << body_indent() << "let " << name << " : " << type << " = ";
  }

  void emit_var(std::string name, std::string type) {
    body_ << body_indent() << "var " << name << " : " << type << " = ";
  }

  const char *get_pointer_int_type_name() {
    return "i32";
  }

  const char *get_primitive_type_name(DataType dt) {
    if (dt->is_primitive(PrimitiveTypeID::f32)) {
      return "f32";
    }
    if (dt->is_primitive(PrimitiveTypeID::i32)) {
      return "i32";
    }
    TI_ERROR("unsupported primitive type {}", dt->to_string());
    return "";
  }

  std::string get_scalar_or_vector_type_name(DataType dt, int num_components) {
    std::string prim_name = get_primitive_type_name(dt);
    std::string type_name = prim_name;
    if (num_components > 1) {
      type_name = std::string("vec") + std::to_string(num_components) + "<" +
                  prim_name + ">";
    }
    return type_name;
  }

  std::string get_scalar_or_vector_expr(const std::vector<Stmt *> &values,
                                        std::string type_name) {
    std::string output_expr = values[0]->raw_name();
    if (values.size() > 1) {
      output_expr = type_name + "(" + values[0]->raw_name();
      for (int i = 1; i < values.size(); ++i) {
        output_expr += ", ";
        output_expr += values[i]->raw_name();
      }
      output_expr += ")";
    }
    return output_expr;
  }

  void generate_serial_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::serial;
    // task_attribs_.buffer_binds = get_common_buffer_binds();
    task_attribs_.advisory_total_num_threads = 1;
    task_attribs_.advisory_num_threads_per_group = 1;

    start_compute_function(1);
    stmt->body->accept(this);
  }

  void generate_range_for_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::range_for;
    // task_attribs_.buffer_binds = get_common_buffer_binds();

    task_attribs_.range_for_attribs = TaskAttributes::RangeForAttributes();
    auto &range_for_attribs = task_attribs_.range_for_attribs.value();
    range_for_attribs.const_begin = stmt->const_begin;
    range_for_attribs.const_end = stmt->const_end;
    range_for_attribs.begin =
        (stmt->const_begin ? stmt->begin_value : stmt->begin_offset);
    range_for_attribs.end =
        (stmt->const_end ? stmt->end_value : stmt->end_offset);
    int block_size = stmt->block_dim;
    task_attribs_.advisory_num_threads_per_group = block_size;
    start_compute_function(block_size);

    std::string total_elems_value;
    std::string begin_expr_value;
    std::string end_expr_value;
    if (range_for_attribs.const_range()) {
      const int num_elems = range_for_attribs.end - range_for_attribs.begin;
      begin_expr_value = std::to_string(stmt->begin_value);
      total_elems_value = std::to_string(num_elems);
      end_expr_value = std::to_string(stmt->begin_value + num_elems);
      task_attribs_.advisory_total_num_threads = num_elems;
    } else {
      if (!stmt->const_begin) {
        emit_let("begin_idx", "i32");
        body_ << std::to_string(stmt->begin_offset / 4) << ";\n";

        std::string gtmps_buffer_member_name =
            get_buffer_member_name(ResourceInfo(ResourceType::GlobalTemps));
        begin_expr_value = gtmps_buffer_member_name + "[begin_idx]";
      } else {
        begin_expr_value = std::to_string(stmt->begin_value);
      }

      if (!stmt->const_end) {
        emit_let("end_idx", "i32");
        body_ << std::to_string(stmt->end_offset / 4) << ";\n";
        std::string gtmps_buffer_member_name =
            get_buffer_member_name(ResourceInfo(ResourceType::GlobalTemps));
        end_expr_value = gtmps_buffer_member_name + "[end_idx]";
      } else {
        end_expr_value = std::to_string(stmt->end_value);
      }
      total_elems_value = "end_ - begin_";
      task_attribs_.advisory_total_num_threads = 65536;
    }
    emit_let("begin_", "i32");
    body_ << begin_expr_value << ";\n";
    emit_let("end_", "i32");
    body_ << body_indent() << end_expr_value << ";\n";
    emit_let("total_elems", "i32");
    body_ << total_elems_value << ";\n";

    emit_let("total_invocs", "i32");
    body_ << block_size << " * i32(n_workgroups.x);\n";

    emit_var("ii", "i32");
    body_ << "i32(gid3.x) + begin_;\n";

    body_ << body_indent() << "loop {\n";
    indent();
    body_ << body_indent() << "if(ii >= end_) { break;  }\n";

    stmt->body->accept(this);

    body_ << body_indent() << "ii = ii + total_invocs;\n";
    dedent();
    body_ << body_indent() << "}\n";
  }

  void generate_vertex_for_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::vertex_for;
    // task_attribs_.buffer_binds = get_common_buffer_binds();
    task_attribs_.advisory_total_num_threads = 1;
    task_attribs_.advisory_num_threads_per_group = 1;

    stmt->body->accept(this);
    emit_graphics_function("vertex");
  }

  void generate_fragment_for_kernel(OffloadedStmt *stmt) {
    task_attribs_.name = task_name_;
    task_attribs_.task_type = OffloadedTaskType::fragment_for;
    // task_attribs_.buffer_binds = get_common_buffer_binds();
    task_attribs_.advisory_total_num_threads = 1;
    task_attribs_.advisory_num_threads_per_group = 1;

    stmt->body->accept(this);
    emit_graphics_function("fragment");
  }

  StringBuilder global_decls_;

  StringBuilder stage_in_struct_begin_;
  StringBuilder stage_in_struct_body_;
  StringBuilder stage_in_struct_end_;

  StringBuilder stage_out_struct_begin_;
  StringBuilder stage_out_struct_body_;
  StringBuilder stage_out_struct_end_;

  StringBuilder function_signature_;
  StringBuilder function_body_prologue_;
  StringBuilder body_;
  StringBuilder function_body_epilogue_;
  StringBuilder function_end_;

  std::string assemble_shader() {
    return global_decls_.getString() +

           stage_in_struct_begin_.getString() +
           stage_in_struct_body_.getString() +
           stage_in_struct_end_.getString() +

           stage_out_struct_begin_.getString() +
           stage_out_struct_body_.getString() +
           stage_out_struct_end_.getString() +

           function_signature_.getString() +
           function_body_prologue_.getString() + body_.getString() +
           function_body_epilogue_.getString() + function_end_.getString();
  }

  void start_compute_function(int block_size_x) {
    TI_ASSERT(function_signature_.getString().size() == 0);
    std::string signature_template =
        R"(

@stage(compute) @workgroup_size(BLOCK_SIZE_X, 1, 1)
fn main(
  @builtin(global_invocation_id) gid3 : vec3<u32>, 
  @builtin(num_workgroups) n_workgroups : vec3<u32>) 
{

)";
    string_replace_all(signature_template, "BLOCK_SIZE_X",
                       std::to_string(block_size_x));
    function_signature_ << signature_template;
    function_end_ << "\n}\n";
  }

  void emit_graphics_function(const std::string &stage_name) {
    TI_ASSERT(function_signature_.getString().size() == 0);
    std::string signature_template =
        R"(

@stage(STAGE_NAME)
fn main(MAYBE_INPUT) MAYBE_OUTPUT 
{

)";
    string_replace_all(signature_template, "STAGE_NAME", stage_name);
    if (stage_in_struct_begin_.getString().size() > 0) {
      string_replace_all(signature_template, "MAYBE_INPUT",
                         "stage_input: StageInput");
    } else {
      string_replace_all(signature_template, "MAYBE_INPUT", "");
    }
    if (stage_out_struct_begin_.getString().size() > 0) {
      string_replace_all(signature_template, "MAYBE_OUTPUT", "-> StageOutput");
    } else {
      string_replace_all(signature_template, "MAYBE_OUTPUT", "");
    }
    function_signature_ << signature_template;
    function_end_ << "\n}\n";

    if (enforce_16bytes_alignment()) {
      std::string helper =
          R"(

fn find_vec4_component(v: vec4<i32>, index: i32) -> i32 
{
  if((index & 3) == 0){
    return v.x;
  }
  if((index & 3) == 1){
    return v.y;
  }
  if((index & 3) == 2){
    return v.z;
  }
  return v.w;
}

)";
      global_decls_ << helper;
    }
  }

  void ensure_stage_in_struct() {
    if (stage_in_struct_begin_.getString().size() > 0) {
      return;
    }
    stage_in_struct_begin_ << "struct StageInput {\n";
    stage_in_struct_end_ << "};\n";
  }

  void ensure_stage_out_struct() {
    if (stage_out_struct_begin_.getString().size() > 0) {
      return;
    }
    stage_out_struct_begin_ << "struct StageOutput {\n";
    stage_out_struct_end_ << "};\n";

    function_body_prologue_ << "  var stage_output: StageOutput;\n";
    function_body_epilogue_ << "  return stage_output;\n";
  }

  std::unordered_set<std::string> stage_in_members_;
  void add_stage_in_member(const std::string &name,
                           const std::string &dt,
                           int loc) {
    if (stage_in_members_.find(name) == stage_in_members_.end()) {
      stage_in_struct_body_ << "  @location(" << std::to_string(loc) << ") "
                            << name << ": " << dt << ";\n";
      stage_in_members_.insert(name);
    }
  }

  std::unordered_set<std::string> stage_out_members_;
  void add_stage_out_member(const std::string &name,
                            const std::string &dt,
                            int loc) {
    if (stage_out_members_.find(name) == stage_out_members_.end()) {
      stage_out_struct_body_ << "  @location(" << std::to_string(loc) << ") "
                             << name << ": " << dt << ";\n";
      stage_out_members_.insert(name);
    }
  }

  std::unordered_set<std::string> stage_out_built_in_members_;
  void add_stage_out_builtin_member(const std::string &name,
                                    const std::string &dt,
                                    const ::std::string &builtin) {
    if (stage_out_built_in_members_.find(name) ==
        stage_out_built_in_members_.end()) {
      stage_out_struct_body_ << "  @builtin(" << builtin << ") " << name << ": "
                             << dt << ";\n";
      stage_out_built_in_members_.insert(name);
    }
  }

  int body_indent_count_ = 1;
  void indent() {
    body_indent_count_++;
  }
  void dedent() {
    body_indent_count_--;
  }
  std::string body_indent() {
    return std::string(body_indent_count_ * 2, ' ');
  }

  int next_internal_temp = 0;
  std::string get_temp(std::string hint = "") {
    return std::string("_internal_temp") +
           std::to_string(next_internal_temp++) + "_" + hint;
  }

  std::unordered_map<std::string, PointerInfo> pointer_infos_;

  bool is_vertex_for_or_fragment_for() {
    return task_ir_->task_type == OffloadedTaskType::vertex_for ||
           task_ir_->task_type == OffloadedTaskType::fragment_for;
  }

  bool enforce_16bytes_alignment() {
    // The issue here is that WebGPU doesn't allow vertex shaders to use storage
    // buffers, and uniform buffer elements must be 16 bytes aligned Also, the
    // vertex shader buffer binding has to be "compatible" with the fragment
    // buffer one, so the fragment must also use 16 bytes aligned uniform buffer
    return is_vertex_for_or_fragment_for();
  }

  std::string get_raw_data_type_name() {
    if (!enforce_16bytes_alignment()) {
      return "i32";
    } else {
      return "vec4<i32>";
    }
  }

  int get_raw_data_type_size() {
    if (!enforce_16bytes_alignment()) {
      return 4;
    } else {
      return 16;
    }
  }

  std::string get_raw_data_index_shift() {
    if (!enforce_16bytes_alignment()) {
      return "";
    } else {
      return " >> 2u";
    }
  }

  std::string get_buffer_member_name(ResourceInfo buffer) {
    return get_buffer_name(buffer) + ".member";
  }

  int divUp(size_t a, int b) {
    if (a % b == 0) {
      return a / b;
    }
    return (a / b) + 1;
  }

  int get_element_count(ResourceInfo buffer) {
    switch (buffer.type) {
      case ResourceType::RootNormal: {
        return divUp(compiled_structs_[buffer.resource_id].root_size,
                     get_raw_data_type_size());
      }
      case ResourceType::RootAtomicI32: {
        return divUp(compiled_structs_[buffer.resource_id].root_size,
                     4);  // WGSL doesn't allow atomic<vec4<i32>>, so the type
                          // size is always 4
      }
      case ResourceType::GlobalTemps: {
        return divUp(
            65536,
            get_raw_data_type_size());  // maximum size allowed by WebGPU Chrome
                                        // DX backend. matches Runtime.ts
      }
      case ResourceType::RandStates: {
        return 65536;  // matches Runtime.ts // note that we have up to 65536
                       // shader invocations
      }
      case ResourceType::Args: {
        return divUp(ctx_attribs_->args_bytes(), get_raw_data_type_size());
      }
      case ResourceType::Rets: {
        return divUp(ctx_attribs_->rets_bytes(), get_raw_data_type_size());
      }
    }
  }

  std::string get_buffer_name(ResourceInfo buffer) {
    std::string name;
    std::string element_type = get_raw_data_type_name();
    switch (buffer.type) {
      case ResourceType::RootNormal: {
        name = "root_buffer_" + std::to_string(buffer.resource_id) + "_";
        break;
      }
      case ResourceType::RootAtomicI32: {
        name = "root_buffer_" + std::to_string(buffer.resource_id) + "_atomic_";
        element_type = "atomic<i32>";
        break;
      }
      case ResourceType::GlobalTemps: {
        name = "global_tmps_";
        break;
      }
      case ResourceType::RandStates: {
        name = "rand_states_";
        element_type = "RandState";
        break;
      }
      case ResourceType::Args: {
        name = "args_";
        break;
      }
      case ResourceType::Rets: {
        name = "rets_";
        break;
      }
      default: {
        TI_ERROR("not a buffer");
      }
    }
    if (task_attribs_.resource_bindings.find(buffer) ==
        task_attribs_.resource_bindings.end()) {
      int binding =
          binding_point_begin_ + task_attribs_.resource_bindings.size();
      task_attribs_.resource_bindings[buffer] = binding;
      int element_count = get_element_count(buffer);
      declare_new_buffer(buffer, name, binding, element_type, element_count);
    }
    return name;
  }

  void declare_new_buffer(ResourceInfo buffer,
                          std::string name,
                          int binding,
                          std::string element_type,
                          int element_count) {
    std::string decl_template =
        R"(

struct BUFFER_TYPE_NAME {
    member: array<ELEMENT_TYPE, ELEMENT_COUNT>;
};
@group(0) @binding(BUFFER_BINDING)
var<STORAGE_AND_ACCESS> BUFFER_NAME: BUFFER_TYPE_NAME;

)";

    string_replace_all(decl_template, "BUFFER_TYPE_NAME", name + "_type");
    string_replace_all(decl_template, "BUFFER_NAME", name);
    string_replace_all(decl_template, "BUFFER_BINDING",
                       std::to_string(binding));
    string_replace_all(decl_template, "ELEMENT_TYPE", element_type);
    string_replace_all(decl_template, "ELEMENT_COUNT",
                       std::to_string(element_count));
    if (!is_vertex_for_or_fragment_for()) {
      string_replace_all(decl_template, "STORAGE_AND_ACCESS",
                         "storage, read_write");
    } else {
      // WGSL vertex shaders are not allowed to use storage buffers... (why?)
      string_replace_all(decl_template, "STORAGE_AND_ACCESS", "uniform");
    }
    global_decls_ << decl_template;
  }

  std::string get_texture_name(ResourceInfo texture_info) {
    if (texture_info.type != ResourceType::Texture &&
        texture_info.type != ResourceType::StorageTexture) {
      TI_ERROR("not a texture");
    }
    bool is_storage_texture = texture_info.type == ResourceType::StorageTexture;
    Texture *texture = textures_.at(texture_info.resource_id);
    std::string name = "texture_" + std::to_string(texture->id) + "_";
    if (is_storage_texture) {
      name += "storage_";
    }
    std::string element_type =
        get_primitive_type_name(texture->params.primitive);
    std::string type_name;
    bool is_depth = texture->params.is_depth;
    switch (texture->params.dimensionality) {
      case TextureDimensionality::Dim2d: {
        if (!is_depth) {
          if (is_storage_texture) {
            type_name = "texture_storage_2d";
          } else {
            type_name = "texture_2d";
          }
          break;
        }
        TI_ERROR("depth texture not supported");
      }

      default: {
        TI_ERROR("unrecgnized dimensionality")
        break;
      }
    }
    if (task_attribs_.resource_bindings.find(texture_info) ==
        task_attribs_.resource_bindings.end()) {
      int binding =
          binding_point_begin_ + task_attribs_.resource_bindings.size();
      task_attribs_.resource_bindings[texture_info] = binding;
      std::string template_args;
      if (is_storage_texture) {
        template_args = "<" + texture->params.format + ", write>";
      } else {
        template_args = "<" + element_type + ">";
      }
      declare_new_texture(texture_info, name, type_name, template_args,
                          binding);
    }
    return name;
  }

  void declare_new_texture(ResourceInfo texture,
                           std::string name,
                           std::string type_name,
                           std::string template_args,
                           int binding) {
    std::string decl_template =
        R"(

@group(0) @binding(TEXTURE_BINDING)
var TEXTURE_NAME: TYPE_NAME TEMPLATE_ARGS;

)";

    string_replace_all(decl_template, "TYPE_NAME", type_name);
    string_replace_all(decl_template, "TEXTURE_NAME", name);
    string_replace_all(decl_template, "TEXTURE_BINDING",
                       std::to_string(binding));
    string_replace_all(decl_template, "TEMPLATE_ARGS", template_args);
    global_decls_ << decl_template;
  }

  std::string get_sampler_name(ResourceInfo sampler_info) {
    if (sampler_info.type != ResourceType::Sampler) {
      TI_ERROR("not a sampler");
    }
    Texture *texture = textures_.at(sampler_info.resource_id);
    std::string name = "sampler_" + std::to_string(texture->id) + "_";
    std::string type_name = "sampler";
    bool is_depth = texture->params.is_depth;
    if (is_depth) {
      TI_ERROR("depth sampler not supported");
    }
    if (task_attribs_.resource_bindings.find(sampler_info) ==
        task_attribs_.resource_bindings.end()) {
      int binding =
          binding_point_begin_ + task_attribs_.resource_bindings.size();
      task_attribs_.resource_bindings[sampler_info] = binding;
      declare_new_sampler(sampler_info, name, type_name, binding);
    }
    return name;
  }

  void declare_new_sampler(ResourceInfo sampler,
                           std::string name,
                           std::string type_name,
                           int binding) {
    std::string decl_template =
        R"(

@group(0) @binding(SAMPLER_BINDING)
var SAMPLER_NAME: TYPE_NAME;

)";

    string_replace_all(decl_template, "TYPE_NAME", type_name);
    string_replace_all(decl_template, "SAMPLER_NAME", name);
    string_replace_all(decl_template, "SAMPLER_BINDING",
                       std::to_string(binding));
    global_decls_ << decl_template;
  }

  bool rand_initiated_ = false;
  void init_rand() {
    if (rand_initiated_) {
      return;
    }
    std::string struct_decl =
        R"(

struct RandState{
  x: u32;
  y: u32;
  z: u32;
  w: u32;
};

)";
    global_decls_ << struct_decl;
    std::string rand_states_member_name =
        get_buffer_member_name(ResourceInfo(ResourceType::RandStates));
    std::string rand_func_decl =
        R"(

fn rand_u32(id: u32) -> u32 {
  var state : RandState = STATES[id];
  if(state.x == 0u && state.y == 0u && state.z == 0u && state.w == 0u){
    state.x = 123456789u * id * 1000000007u;
    state.y = 362436069u;
    state.z = 521288629u;
    state.w = 88675123u;
  }
  let t : u32 = state.x ^ (state.x << 11u);
  state.x = state.y;
  state.y = state.z;
  state.z = state.w;
  state.w = (state.w ^ (state.w >> 19u)) ^ (t ^ (t >> 8u)); 
  let result : u32 = state.w * 1000000007u;
  STATES[id] = state;
  return result;
}

fn rand_f32(id:u32) -> f32 {
  let u32_res : u32 = rand_u32(id);
  return f32(u32_res) * (1.0f / 4294967296.0f);
}

fn rand_i32(id:u32) -> i32 {
  let u32_res : u32 = rand_u32(id);
  return i32(u32_res);
}

)";
    string_replace_all(rand_func_decl, "STATES", rand_states_member_name);
    global_decls_ << rand_func_decl;
    rand_initiated_ = true;
  }

  OffloadedStmt *const task_ir_;  // not owned
  std::vector<CompiledSNodeStructs> compiled_structs_;
  std::unordered_map<int, Texture *> textures_;
  std::unordered_map<int, int> snode_to_root_;
  const KernelContextAttributes *const ctx_attribs_;  // not owned
  const std::string task_name_;

  // vertex shader and fragment shaders share their buffer bindings
  // to avoid conflict, the fisrt binding used by frag shader should be
  // vert.bindings.size()
  int binding_point_begin_;

  TaskAttributes task_attribs_;
  std::unordered_map<int, GetRootStmt *>
      root_stmts_;  // maps root id to get root stmt
};

}  // namespace

KernelCodegen::KernelCodegen(const Params &params)
    : params_(params), ctx_attribs_(*params.kernel) {
}

void KernelCodegen::run(TaichiKernelAttributes &kernel_attribs,
                        std::vector<std::string> &generated_wgsl) {
  auto *root = params_.kernel->ir->as<Block>();
  auto &tasks = root->statements;
  // vertex shader and fragment shaders share their buffer bindings
  // to avoid conflict, the fisrt binding used by frag shader should be
  // vert.bindings.size()
  int next_binding_point_begin = 0;
  for (int i = 0; i < tasks.size(); ++i) {
    TaskCodegen::Params tp;
    tp.task_ir = tasks[i]->as<OffloadedStmt>();
    tp.task_id_in_kernel = i;
    tp.compiled_structs = params_.compiled_structs;
    tp.textures = params_.textures;
    tp.ctx_attribs = &ctx_attribs_;
    tp.ti_kernel_name = params_.ti_kernel_name;
    tp.binding_point_begin = next_binding_point_begin;

    TaskCodegen cgen(tp);

    auto task_res = cgen.run();

    if (tp.task_ir->task_type == OffloadedTaskType::vertex_for) {
      next_binding_point_begin = task_res.task_attribs.resource_bindings.size();
    } else {
      next_binding_point_begin = 0;
    }

    kernel_attribs.tasks_attribs.push_back(std::move(task_res.task_attribs));
    generated_wgsl.push_back(std::move(task_res.wgsl_code));
  }
  kernel_attribs.ctx_attribs = std::move(ctx_attribs_);
  kernel_attribs.name = params_.ti_kernel_name;
  kernel_attribs.is_jit_evaluator = params_.kernel->is_evaluator;
}

void lower(Kernel *kernel) {
  auto &config = kernel->program->config;
  config.demote_dense_struct_fors = true;
  irpass::compile_to_executable(kernel->ir.get(), config, kernel, kernel->grad,
                                /*ad_use_stack=*/false, config.print_ir,
                                /*lower_global_access=*/true,
                                /*make_thread_local=*/false);
}

}  // namespace webgpu
}  // namespace lang
}  // namespace taichi
