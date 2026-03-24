"""
gRPC Server — запуск gRPC сервера параллельно с REST.

Запуск:
    python -m app.grpc_api.server          # Только gRPC (порт 50051)
    python main.py                          # REST + gRPC (оба вместе)
"""

import asyncio
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

from app.grpc_api import rag_pb2, rag_pb2_grpc
from app.grpc_api.service import RAGServicer
from app.config import settings
from app.utils.logging import get_logger

log = get_logger("grpc_server")

# gRPC порт (REST port + 1 по умолчанию, или настройка)
GRPC_PORT = settings.grpc_port


async def serve_grpc(port: int = None) -> grpc.aio.Server:
    """Запустить gRPC сервер.

    Args:
        port: Порт для gRPC (default: 50051)

    Returns:
        Запущенный gRPC сервер
    """
    port = port or GRPC_PORT

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )

    # Регистрируем сервис
    rag_pb2_grpc.add_RAGServiceServicer_to_server(RAGServicer(), server)

    # Reflection — для grpcurl и других инструментов
    service_names = (
        rag_pb2.DESCRIPTOR.services_by_name["RAGService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"0.0.0.0:{port}"
    server.add_insecure_port(listen_addr)

    await server.start()

    log.info("═══════════════════════════════════════")
    log.info("  gRPC сервер запущен: {}", listen_addr)
    log.info("  Reflection: включён (grpcurl ready)")
    log.info("═══════════════════════════════════════")

    return server


async def main():
    """Standalone запуск gRPC сервера."""
    server = await serve_grpc()
    log.info("gRPC сервер ожидает запросы на порту {}...", GRPC_PORT)

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        log.info("Остановка gRPC сервера...")
        await server.stop(5)


if __name__ == "__main__":
    asyncio.run(main())
